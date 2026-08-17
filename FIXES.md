# Engineering log

The first draft of this notebook ran end-to-end in my head and nowhere else. This file is the log of every defect I found taking it from "written" to "training", grouped by how expensive each one was to catch.

The interesting class is the first one: bugs that do not raise, do not warn, and still produce a falling loss curve. Those are the reason the [verification checks](README.md#verification) exist.

---

## 1. Silent correctness bugs

These trained. They just trained the wrong model.

### RoPE was rotating across heads instead of positions

```python
# before — rotary receives (B, H, T, D)
q = self.rotary(q.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
```

`Rotary.forward` reads the sequence length from `dim=-3` and broadcasts `cos`/`sin` as `[None, :T, None, :]` — it expects `(B, T, H, D)`. Feeding it the transposed layout applies the position-dependent rotation along the **head** axis: every position gets the same rotation, and heads get rotated as if they were timesteps. Loss still falls (the model leans on content alone), so nothing looks wrong.

**Fix:** apply rotary in `(B, T, H, D)` and transpose to `(B, H, T, D)` afterwards.
**Test:** with identical content at every position, `⟨y_i, y_j⟩` must depend only on `j − i`.

### Grouped-query attention was mis-wired

```python
# before
self.n_kv_heads = self.n_heads // self.n_kv_heads   # 8 // 4 = 2, field overwritten
```

`__post_init__` overwrote the number of KV heads with the *group* count, and `config.n_kv_groups` — which `Qwen3Attention` reads — was never defined at all. The model built 2 KV heads instead of 4 and could not construct attention without an `AttributeError` on the group count.

**Fix:** derive `n_kv_groups = n_heads // n_kv_heads` as a separate attribute; keep `n_kv_heads` meaning what its name says. Divisibility assertions moved above the derived values.

### The step counter counted micro-batches

`step` incremented once per micro-batch while the LR scheduler, the eval trigger and `max_steps` all treated it as an optimizer update. With `gradient_accumulation_steps=4` that meant 125 real updates and a cosine schedule that only ever traversed 25% of its curve — the model stopped mid-decay at a still-high LR.

**Fix:** one `step` = one optimizer update. Micro-batches are pulled from an endless iterator inside the step.

### Muon never updated anything correctly

Five separate defects in one `step()`:

| Line | Problem |
|---|---|
| `for group in self.params_groups` | typo — `AttributeError` on the first call |
| `if 'momentum_buffer' not in state: buf = state['momentum_buffer']` | reads the key it is supposed to create — `KeyError` |
| `buf.lerp(g, ...)` | not in-place; the momentum buffer never accumulated |
| `g` | undefined name — the gradient is `p.grad` |
| `zeropowervia_newtonschulz5(...)` | misspelled call into the Newton–Schulz function |

Plus `assert G.dim >= 2`, which compares a bound *method* to an int and is therefore always true.

**Fix:** rewrote `step()` — buffer initialized with `zeros_like`, in-place `lerp_`, Nesterov applied to `p.grad`, `assert G.ndim >= 2`.
**Test:** every matrix parameter moves after one step and stays finite; the orthogonalized update's singular values land in ≈ [0.69, 1.10].

### Newton–Schulz in bf16 on a T4

The reference Muon implementation casts to bfloat16. The target hardware here is a T4 (SM 7.5), which has no native bf16.

**Fix:** run the iteration in fp32 and cast the update back to the parameter dtype on application. Also dropped `@torch.compile` from the function — recompilation cost on Colab outweighed the gain at this scale.

### `deterministic` and `benchmark` both set to `True`

Mutually exclusive requests: `benchmark=True` picks kernels by timing, which is exactly the nondeterminism `deterministic=True` asks cuDNN to avoid.

**Fix:** `deterministic=True`, `benchmark=False` — reproducibility is the point of a function named `set_seed`.

### Evaluation never fired

`eval_every=500` with `max_steps=500`, and the eval guarded by `step > 0`. The loop exits before step 500, so no mid-training validation ever ran.

**Fix:** `eval_every=100` — five checkpoints across the run, which is where the results table comes from.

---

## 2. Crashes

### `load_and_cache_data` returned before doing any work

The cache-miss path was written *after* an unconditional `return`, so the function only ever tried the cache branch — and referenced `cached_data` before assignment when the cache did not exist. That is the `UnboundLocalError` the first run died on.

Underneath it, four more defects in the code that had never executed:

- `return text, tokenizer, tokens` — `text` is the loop variable, not the `texts` list
- `tokens = all+tokens[:config.max_tokens]` — parses as `all + tokens[...]`, i.e. the builtin `all` plus a list
- `all_tokens.append(tokens)` built a list of lists; `TextTokenDataset` expects one flat stream
- `config.vocab_size` on its own line — reads an attribute and discards it; the model's vocabulary size was never set on the cache-miss path
- `AutoTokenizer.from_pretrained("HuggingfaceTB/...")` — wrong capitalization, 404 on the Hub

**Fix:** one linear function — cache hit returns early, cache miss streams, tokenizes with `extend`, truncates, sets `vocab_size`, writes the cache.

### `MinimalLLM.forward` called two attributes that did not exist

`self.position_dropout(x)` and `self.lm_head(x)`. The class defined `self.output_dropout` and `self.lm_head_weight` — a tied weight tensor, not a module.

**Fix:** one `input_dropout`, and an explicit tied head via `F.linear(x, self.token_embedding.weight)`. Also deleted the unused learned `position_embedding` — with RoPE inside attention it was dead weight in both senses.

### `repeat_kv` parameter name typo

Signature took `hidden_States`; the body used `hidden_states`. Every call raised `NameError`.

---

## 3. API and hygiene

- **`torch.cuda.amp` → `torch.amp`.** The old `autocast()` / `GradScaler()` entry points are deprecated; the device-typed API replaces them.
- **Collapsed the duplicated AMP branch.** `GradScaler(enabled=config.use_amp)` no-ops when disabled, so the fp16 and fp32 paths are now one block instead of two that can drift apart.
- **Removed the `try`/`except` double-load in `load_trained_model`.** Registering `ModelConfig` via `add_safe_globals` makes a single `weights_only=True` load correct; catching the failure and retrying with `weights_only=False` defeats the point of the flag.
- **Generation could not exceed the context window.** No cropping, so past 512 tokens the rotary buffer index went out of range. Now feeds `generated_ids[:, -max_seq_len:]`; `max_length` renamed to `max_new_tokens`, which is what it counts.
- **Top-k rewritten as a threshold** (`logits < kth_value → -inf`) instead of building a full `-inf` tensor and scattering into it.
- **`print("Set all seeds to {seed}")`** — missing `f` prefix, printing the literal braces.
- **`torch.cat((y1, y2), 3)`** — hardcoded axis in `Rotary`; now `dim=-1`.
- **Config naming.** `gradient_accum_steps` in the dataclass, `config.gradient_accumulation_steps` in the training loop. Unified on the long name.
- **Dead config.** `sliding_window` was never read — SDPA is called with `is_causal=True` and no window. Removed rather than left as an implied feature.
- **DataLoader.** Added `drop_last=True` (ragged final batch), `num_workers=2`, and `pin_memory` gated on CUDA.
- **Reported metrics.** Gradient norm added to the progress bar; training wall-clock was computed and then never printed.

---

## How these were found

A harness that executes every notebook cell, then asserts on behaviour rather than on absence of exceptions:

1. Config invariants — `d_k`, `n_kv_heads`, `n_kv_groups` are what they claim to be
2. `repeat_kv` maps each KV head onto its query group
3. RoPE relative-position invariance (catches the axis bug)
4. Causality — a later token cannot change earlier logits
5. Forward/backward shapes and finiteness
6. Newton–Schulz singular-value spread
7. One Muon step moves every matrix parameter
8. Full training loop, with and without AMP
9. Checkpoint round-trip under `weights_only=True`
10. Generation, including past the context window

All ten run on CPU in under a minute with a randomly generated token stream — no GPU, no dataset download — which is what makes it practical to run them before spending T4 time.

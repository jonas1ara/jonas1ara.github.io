---
title: "TorchSharp LLaMA in F#"
description: "A from-scratch implementation of Llama 3.2 inference in F#: Grouped Query Attention, RoPE, SwiGLU, KV cache, and a Tiktoken tokenizer — running on .NET 10 with TorchSharp."
Author: Jonas Lara
date: 2026-03-13 00:00:00 +0000
categories: [Functional Programming, F#, Machine Learning]
tags: [fsharp, llama, transformers, dotnet, torchsharp, machine learning, llm]
image:
  path: /assets/img/post/TorchSharp-LLaMA/llama.jpg
  lqip: https://raw.githubusercontent.com/jonas1ara/jonas1ara.github.io/refs/heads/main/assets/img/post/TorchSharp-LLaMA/llama.jpg
  alt: Loot Llama from Fortnite, representing the Llama model in this post.
---

# Llama.fs

Most LLM inference tools require Python — either directly through PyTorch/Transformers, or indirectly through a sidecar process like Ollama. If you want to run a language model inside a .NET application, your options have historically been limited to HTTP wrappers around external runtimes.

[Llama.fs](https://github.com/jonas1ara/Llama.fs) changes that. It's a from-scratch implementation of the [Llama 3.2](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/) inference engine in F# on .NET 10, using [TorchSharp](https://github.com/dotnet/TorchSharp) for tensor operations and [TorchSharp.PyBridge](https://github.com/dotnet/TorchSharp/tree/main/src/TorchSharp.PyBridge) to load the original `.pth` checkpoint directly into the F# model. The result is a self-contained .NET executable that loads Llama weights and generates text — no Python process, no Ollama, no HTTP, no external runtime.

This post walks through every layer of the implementation: how the tokenizer encodes text, how RoPE positions tokens in complex space, why GQA uses fewer KV heads, what SwiGLU's three matrices do, and how the KV cache makes autoregressive generation efficient.

---

## Project Structure

The implementation follows a clean separation of concerns across five files:

| File | Responsibility |
|---|---|
| `Utils.fs` | RoPE frequency precomputation, rotary embedding application, KV-head repeat |
| `Tokenizer.fs` | Tiktoken BPE tokenizer with special token handling for the Llama 3 instruct template |
| `Model.fs` | Full transformer architecture: `RMSNorm`, `SelfAttention` (GQA), `FeedForward` (SwiGLU), `EncoderBlock`, `Transformer` |
| `Llama.fs` | Model loading via `TorchSharp.PyBridge`, KV-cache generation loop, top-p sampling |
| `Program.fs` | Interactive CLI loop with the Llama 3 instruct template |

Each file can be read in isolation. `Utils.fs` has no dependencies on the rest. `Tokenizer.fs` is completely independent of the model. `Model.fs` only imports `Utils.fs`. `Llama.fs` orchestrates everything.

---

## Utils.fs: Rotary Positional Embeddings

### Why Positional Embeddings At All?

A transformer's attention mechanism is **permutation-invariant** — if you shuffle the tokens in the input, the attention scores change but the model has no built-in way to know the tokens moved. Two sentences with the same words in different orders look identical to a transformer without positional information.

Positional embeddings solve this by injecting position-dependent information into each token's representation before attention is computed. Earlier models (BERT, GPT-2) added a learned or fixed sinusoidal vector to each token embedding. Llama uses a different approach: **RoPE** (Rotary Positional Embeddings), which encodes position as a *rotation applied to the query and key vectors* inside each attention head.

The key advantage of RoPE over additive embeddings is that the dot product `Q · K` naturally captures the *relative distance* between positions — if token `m` and token `n` are encoded with rotations `θm` and `θn`, their dot product depends only on `|m - n|`. This lets the model generalize to sequence lengths longer than those seen during training.

### Precomputing the Rotation Frequencies

Each pair of dimensions `(2i, 2i+1)` in a head gets its own rotation frequency. The frequency for pair `i` in a head of dimension `d` is:

```
θᵢ = 1 / (rope_theta ^ (2i / d))
```

For Llama 3.2, `rope_theta = 500,000` (much larger than the original 10,000 — this extends the effective context length). The result is a set of frequencies that decay geometrically: the first pairs of dimensions rotate fast (high frequency, captures short-range patterns), the last pairs rotate slowly (low frequency, captures long-range patterns).

These frequencies are multiplied by position indices `m = [0, 1, 2, ..., seqLen-1]` using an outer product, producing a `[seqLen, headDim/2]` matrix of angles. `torch.polar` converts each angle into a unit complex number `e^(iθ)`:

```fsharp
let precomputeThetaPosFrequencies (headDim: int) (seqLen: int) (theta: float32) : Tensor =
    if headDim % 2 <> 0 then invalidArg "headDim" "Dimension must be divisible by 2"
    // indices = [0, 2, 4, ..., headDim-2]
    let indices    : Tensor = torch.arange(0, headDim, 2).``to``(torch.float32)
    // thetaInput[i] = theta ^ (-indices[i] / headDim)  →  decaying frequencies
    let thetaInput : Tensor = (indices / f32 headDim).mul(-(log theta)).exp()
    // m = [0, 1, 2, ..., seqLen-1]
    let m          : Tensor = torch.arange(seqLen)
    // freqs[m, i] = m * thetaInput[i]  →  outer product gives all (position, frequency) pairs
    let freqs      : Tensor = torch.outer(m.``to``(torch.float32), thetaInput)
    // polar(1, θ) = cos(θ) + i·sin(θ)  →  unit complex number at angle θ
    torch.polar(torch.ones_like(freqs), freqs)
```

The result is a `[seqLen, headDim/2]` tensor of complex numbers. Each entry is the rotation to apply to one dimension pair at one position.

### Applying the Rotation

To rotate a query or key tensor, each pair of real-valued dimensions `(x₁, x₂)` is treated as a complex number `x₁ + i·x₂` and multiplied by the corresponding rotation `e^(iθ)`. Complex multiplication handles the 2D rotation automatically:

```
(x₁ + i·x₂) · (cos θ + i·sin θ) = (x₁ cos θ − x₂ sin θ) + i·(x₁ sin θ + x₂ cos θ)
```

In code, the tensor is reshaped to expose pairs as a complex dimension, then `view_as_complex` reinterprets consecutive float pairs as single complex numbers — no data is copied, it's just a view change:

```fsharp
let applyRotaryEmbeddings (input: Tensor) (freqsComplex: Tensor) : Tensor =
    // Cast to float32 because complex ops require float32
    // Reshape [B, S, H, D] → [B, S, H, D/2, 2]  (expose pairs)
    let inputComplex =
        input.to_type(ScalarType.Float32)
             .reshape(input.shape.[0], input.shape.[1], input.shape.[2], -1L, 2L)
             .view_as_complex()   // [B, S, H, D/2] of complex64
    // freqsComplex is [S, D/2]; unsqueeze adds batch and head dims for broadcasting
    let rotated = (inputComplex * freqsComplex.unsqueeze(0).unsqueeze(2)).view_as_real()
    // Flatten back: [B, S, H, D/2, 2] → [B, S, H, D]
    // type_as restores BFloat16
    rotated.reshape(rotated.shape.[0], rotated.shape.[1], rotated.shape.[2], -1L).type_as(input)
```

RoPE is applied to queries and keys — not to values. Values are never used in the similarity computation, so positional information there is not needed.

### Repeating KV Heads for GQA

Grouped Query Attention (explained in detail in the `SelfAttention` section) uses fewer key/value heads than query heads. Before the matrix multiply that computes attention scores, the KV heads must be expanded to match the query head count.

In Llama 3.2 1B: 32 query heads share 8 KV heads, so each KV head must be repeated 4 times (`nRep = 4`):

```fsharp
let repeatKV (x: Tensor) (nRep: int) : Tensor =
    if nRep = 1 then x   // standard MHA — nothing to do
    else
        let b, s, h, d = x.shape.[0], x.shape.[1], x.shape.[2], x.shape.[3]
        // [B, S, H_kv, D] → [B, S, H_kv, 1, D] → [B, S, H_kv, nRep, D] → [B, S, H_q, D]
        x.unsqueeze(3).expand(b, s, h, i64 nRep, d).reshape(b, s, h * i64 nRep, d)
```

`expand` is a zero-copy broadcast — it creates a view that looks like the tensor is repeated, without allocating new memory. `reshape` then collapses the `H_kv × nRep` dimensions into a single `H_q` dimension. The resulting tensor is what the attention computation consumes.

---

## Model.fs: The Transformer Architecture

### ModelArgs: What the Config File Tells Us

Before loading any weights, the model reads `params.json` from the checkpoint folder. This JSON file contains all the hyperparameters that define the architecture. F# maps them using `[<JsonPropertyName>]` attributes so the PascalCase F# properties match the snake_case JSON keys exactly:

```fsharp
type ModelArgs() =
    [<JsonPropertyName("dim")>]                member val Dim              : int     = 4096   with get, set
    [<JsonPropertyName("n_layers")>]           member val NLayers          : int     = 32     with get, set
    [<JsonPropertyName("n_heads")>]            member val NHeads           : int     = 32     with get, set
    [<JsonPropertyName("n_kv_heads")>]         member val NKVHeads         : Nullable<int> = Nullable() with get, set
    [<JsonPropertyName("vocab_size")>]         member val VocabSize        : int     = -1     with get, set
    [<JsonPropertyName("multiple_of")>]        member val MultipleOf       : int     = 256    with get, set
    [<JsonPropertyName("ffn_dim_multiplier")>] member val FFNDimMultiplier : Nullable<float32> = Nullable() with get, set
    [<JsonPropertyName("norm_eps")>]           member val NormEps          : float32 = 1e-5f  with get, set
    [<JsonPropertyName("rope_theta")>]         member val RopeTheta        : float32 = 500000f with get, set
    [<JsonPropertyName("max_batch_size")>]     member val MaxBatchSize     : int     = 3     with get, set
    [<JsonPropertyName("max_seq_len")>]        member val MaxSeqLen        : int     = 1024  with get, set
    member _.Dtype = ScalarType.BFloat16
```

For **Llama 3.2 1B Instruct**, the actual `params.json` contains:

| Parameter | Value | What it means |
|---|---|---|
| `dim` | 2048 | Hidden state size — the width of every tensor flowing through the model |
| `n_layers` | 16 | Number of stacked transformer blocks |
| `n_heads` | 32 | Number of query attention heads |
| `n_kv_heads` | 8 | Number of key/value heads (GQA: 4× fewer than query heads) |
| `vocab_size` | 128256 | Number of tokens the model knows |
| `norm_eps` | 1e-5 | Small constant to prevent division by zero in RMSNorm |
| `rope_theta` | 500000 | Base frequency for RoPE — large value extends context length |
| `multiple_of` | 256 | FFN hidden size rounded to this multiple for hardware efficiency |

`NKVHeads` and `FFNDimMultiplier` are `Nullable` because not all Llama variants include them in `params.json` — the code falls back to defaults when they're absent.

---

### RMSNorm: Normalization Without the Mean

Llama replaces LayerNorm with **RMSNorm**. Standard LayerNorm subtracts the mean and divides by the standard deviation. RMSNorm only divides by the root-mean-square — no mean subtraction, no bias parameter. This is simpler, faster, and empirically achieves the same quality.

The formula:
```
RMSNorm(x) = x / RMS(x) · γ    where RMS(x) = √(mean(x²) + ε)
```

`γ` is a learned per-dimension scale (initialized to ones). `ε` (`norm_eps = 1e-5`) prevents the denominator from reaching zero when all activations are near zero.

```fsharp
type RMSNorm(args: ModelArgs) =
    inherit Module<Tensor, Tensor>("RMSNorm")
    // γ: learned scale, one value per dimension, initialized to 1.0
    let weight = Parameter(torch.ones(i64 args.Dim, dtype = args.Dtype))
    do base.RegisterComponents()
    // x.mul(x) = x², mean over last dim, rsqrt = 1/√(·)
    let norm (x: Tensor) = x * torch.rsqrt(x.mul(x).mean([|-1L|], keepdim = true) + args.NormEps)
    // Cast to float32 for the norm computation (BFloat16 lacks precision for squaring),
    // then cast back to BFloat16 with type_as before applying the scale
    override _.forward(input) = weight * (norm (input.to_type(ScalarType.Float32))).type_as(input)
```

The float32 cast is important: squaring BFloat16 values and summing them loses precision because BFloat16 only has 7 mantissa bits. The norm is computed in float32, then the result is cast back to BFloat16 before the scale multiplication. `type_as(input)` preserves the original dtype without hardcoding it.

RMSNorm is used **twice per block** — once before attention and once before the feedforward layer (pre-normalization), as opposed to post-normalization used in older architectures. Pre-norm makes training more stable at large scale.

---

### SelfAttention: Grouped Query Attention

#### The Four Projection Matrices

Standard multi-head attention projects the input into queries, keys, and values using three weight matrices of equal size. Every head gets its own slice. **Grouped Query Attention (GQA)** breaks the symmetry: queries still use one matrix per head, but keys and values share matrices across groups of query heads.

In Llama 3.2 1B:
- `wq`: projects to `n_heads × headDim = 32 × 64 = 2048` dimensions — one full set per query head
- `wk`, `wv`: project to `n_kv_heads × headDim = 8 × 64 = 512` dimensions — 4× smaller
- `wo`: projects the concatenated output back to `dim = 2048`

Why fewer KV heads? The KV cache stores every key and value produced for every token in the sequence. With 32 KV heads at float16, a single layer's cache for a 512-token sequence takes `2 × 32 × 512 × 64 × 2 bytes = 4 MB`. With 8 KV heads it's `1 MB`. Across 16 layers, that difference is 48 MB per inference — meaningful when you're memory-constrained.

```fsharp
type SelfAttention(args: ModelArgs) =
    inherit Module<Tensor, int, Tensor, Tensor, Tensor>("SelfAttention")

    let nKVHeads = if args.NKVHeads.HasValue then args.NKVHeads.Value else args.NHeads
    let nHeadsQ  = args.NHeads
    let nRep     = nHeadsQ / nKVHeads   // 32 / 8 = 4: each KV head serves 4 query heads
    let headDim  = args.Dim / args.NHeads  // 2048 / 32 = 64 dimensions per head

    let wq = Linear(i64 args.Dim, i64(nHeadsQ  * headDim), hasBias = false, dtype = args.Dtype)
    let wk = Linear(i64 args.Dim, i64(nKVHeads * headDim), hasBias = false, dtype = args.Dtype)
    let wv = Linear(i64 args.Dim, i64(nKVHeads * headDim), hasBias = false, dtype = args.Dtype)
    let wo = Linear(i64(nHeadsQ  * headDim), i64 args.Dim, hasBias = false, dtype = args.Dtype)

    // KV cache: pre-allocated tensors that hold all keys/values seen so far
    // Shape: [maxBatch, maxSeqLen, n_kv_heads, headDim]
    let mutable ck = torch.zeros(i64 args.MaxBatchSize, i64 args.MaxSeqLen, i64 nKVHeads, i64 headDim, dtype = args.Dtype)
    let mutable cv = torch.zeros(i64 args.MaxBatchSize, i64 args.MaxSeqLen, i64 nKVHeads, i64 headDim, dtype = args.Dtype)

    do base.RegisterComponents()
```

`ck` and `cv` are `mutable` because they start on CPU and need to be moved to CUDA lazily on the first forward pass — you can't move them in the constructor because the device isn't known yet:

```fsharp
    override _.forward(input, startPos, freqsComplex, mask) =
        // Lazy device migration: move cache to match input device on first CUDA call
        if ck.device <> input.device then
            ck <- ck.``to``(input.device)
            cv <- cv.``to``(input.device)
```

#### The Attention Forward Pass Step by Step

```fsharp
        let bs  = toInt input.shape.[0]   // batch size
        let sl  = toInt input.shape.[1]   // sequence length of *this* input (1 during generation)

        // Step 1: Linear projections
        // wq projects all tokens to query space, then reshape to expose heads
        // [B, S, dim] → [B, S, n_heads, headDim]
        let xq = wq.forward(input).view(i64 bs, i64 sl, i64 nHeadsQ,  i64 headDim)
        let xk = wk.forward(input).view(i64 bs, i64 sl, i64 nKVHeads, i64 headDim)
        let xv = wv.forward(input).view(i64 bs, i64 sl, i64 nKVHeads, i64 headDim)

        // Step 2: Apply RoPE to Q and K — injects position information
        // V is not rotated: values don't participate in similarity computation
        let xq' = Utils.applyRotaryEmbeddings xq freqsComplex
        let xk' = Utils.applyRotaryEmbeddings xk freqsComplex

        // Step 3: Write new keys/values into the cache at positions [startPos, startPos+sl)
        // narrow(dim, start, length) = slice without copying
        // copy_ writes in-place into the pre-allocated cache tensor
        ck.narrow(0, 0L, i64 bs).narrow(1, i64 startPos, i64 sl).copy_(xk') |> ignore
        cv.narrow(0, 0L, i64 bs).narrow(1, i64 startPos, i64 sl).copy_(xv)  |> ignore

        // Step 4: Read ALL keys/values from cache (positions 0 to startPos+sl)
        // During generation, startPos grows each step — we read more context each time
        let keys = Utils.repeatKV (ck.narrow(0, 0L, i64 bs).narrow(1, 0L, i64(startPos + sl))) nRep
        let vals = Utils.repeatKV (cv.narrow(0, 0L, i64 bs).narrow(1, 0L, i64(startPos + sl))) nRep

        // Step 5: Scaled dot-product attention
        // xq'.transpose(1,2): [B, n_heads, S_q, headDim]
        // keys.transpose(1,2).transpose(2,3): [B, n_heads, headDim, S_kv]
        // matmul → scores: [B, n_heads, S_q, S_kv]
        // Dividing by √headDim prevents the dot products from growing too large (which
        // would push softmax into regions of near-zero gradient)
        let scores = torch.matmul(xq'.transpose(1,2), keys.transpose(1,2).transpose(2,3)) / sqrtf(f32 headDim)
        // Add causal mask: −∞ for future positions → softmax → 0 attention weight
        let scores = if isNull mask then scores else scores + mask
        // softmax turns scores into probabilities, matmul with values produces output
        let out    = torch.matmul(functional.softmax(scores, dim = -1L), vals.transpose(1,2))

        // Step 6: Merge heads and project back to model dim
        // [B, n_heads, S, headDim] → [B, S, n_heads * headDim] → [B, S, dim]
        wo.forward(out.transpose(1,2).contiguous().view(i64 bs, i64 sl, -1L))
```

`startPos` is the position of the *first token in the current input* within the full sequence. During prefill (processing the prompt), `startPos = 0` and `sl = promptLength`. During generation, `startPos = previousLength` and `sl = 1`. The cache always holds positions `0..startPos+sl-1` after each step.

---

### FeedForward: SwiGLU Activation

Each transformer block has an attention sublayer and a feedforward sublayer. The feedforward network (FFN) is a simple MLP that transforms each token's representation independently.

Standard transformers use `FFN(x) = W₂ · ReLU(W₁ · x)` — two matrices with ReLU in the middle. Llama uses **SwiGLU**, which introduces a third matrix and replaces ReLU with a gated activation:

```
FFN(x) = W₂ · (SiLU(W₁ · x) ⊙ W₃ · x)
```

The gating mechanism works as follows:
- `W₁ · x` is the "gate" path — passed through `SiLU(x) = x · σ(x)`, which produces smooth, non-zero gradients everywhere (unlike ReLU which is zero for negative inputs)
- `W₃ · x` is the "value" path — computed in parallel
- The element-wise product `⊙` makes each dimension of the hidden state modulate the other, giving the network fine-grained control over information flow

The hidden dimension is computed as `2/3 × 4 × dim`, rounded up to a multiple of 256. For `dim=2048` this gives `hiddenDim = 5632`:

```fsharp
type FeedForward(args: ModelArgs) =
    inherit Module<Tensor, Tensor>("FeedForward")
    let hiddenDim =
        // Base: 4 × dim (standard FFN ratio), then 2/3 of that (SwiGLU adjustment for similar param count)
        let h = 2 * (args.Dim * 4) / 3
        // Optional per-model multiplier from params.json
        let h = if args.FFNDimMultiplier.HasValue then Convert.ToInt32(args.FFNDimMultiplier.Value * f32 h) else h
        // Round up to nearest multiple of 256 for hardware-friendly tensor shapes
        args.MultipleOf * ((h + args.MultipleOf - 1) / args.MultipleOf)
    let w1 = Linear(i64 args.Dim, i64 hiddenDim, hasBias = false, dtype = args.Dtype)
    let w2 = Linear(i64 hiddenDim, i64 args.Dim, hasBias = false, dtype = args.Dtype)
    let w3 = Linear(i64 args.Dim, i64 hiddenDim, hasBias = false, dtype = args.Dtype)
    do base.RegisterComponents()
    // SwiGLU: w2(silu(w1(x)) * w3(x))
    override _.forward(x) = w2.forward(functional.silu(w1.forward(x)) * w3.forward(x))
```

All three linear layers have no bias (`hasBias = false`) — consistent with the rest of the Llama architecture where biases are omitted throughout.

---

### EncoderBlock: Pre-Norm Residual Connections

Each of the 16 transformer blocks in Llama 3.2 1B follows a **pre-normalization** pattern: RMSNorm is applied to the input *before* each sublayer, and the original (un-normalized) input is added back as a residual *after*:

```
output = input + sublayer(RMSNorm(input))
```

This is different from the original "post-norm" transformer where normalization happens after the residual addition. Pre-norm is more stable during training of deep models because the residual path is never normalized — gradients flow directly back through the `+ x` skip connection without going through any normalization layer.

```fsharp
type EncoderBlock(args: ModelArgs) =
    inherit Module<Tensor, int, Tensor, Tensor, Tensor>("EncoderBlock")
    let attention      = new SelfAttention(args)
    let feed_forward   = new FeedForward(args)
    let attention_norm = new RMSNorm(args)   // normalizes BEFORE attention
    let ffn_norm       = new RMSNorm(args)   // normalizes BEFORE feedforward
    do base.RegisterComponents()
    override _.forward(x, pos, freqs, mask) =
        // Attention sublayer: normalize input, run attention, add residual
        let h = attention.forward(attention_norm.forward(x), pos, freqs, mask) + x
        // FFN sublayer: normalize attention output, run FFN, add residual
        feed_forward.forward(ffn_norm.forward(h)) + h
```

Four parameters per block: `attention_norm.weight`, `ffn_norm.weight` (both RMSNorm scales), plus all the linear weights inside `SelfAttention` and `FeedForward`. TorchSharp's `RegisterComponents()` call ensures all sub-modules and their parameters are tracked for checkpoint loading and device migration.

### Transformer: Embedding, Causal Mask, and Output

The top-level `Transformer` handles the beginning and end of the forward pass: token embedding at the start, and the language model head (a linear projection to vocabulary logits) at the end. It also builds the causal attention mask and selects the correct slice of precomputed RoPE frequencies:

```fsharp
override _.forward(tokens, startPos) =
    let sl  = toInt tokens.shape.[1]
    // Convert token IDs to dense vectors: [B, S] → [B, S, dim]
    let h   = tok_embeddings.forward(tokens)
    // Slice the frequencies for positions [startPos, startPos+sl)
    let f   = freqs.[TensorIndex.Slice(startPos, startPos + sl)].``to``(h.device)

    // Build the causal mask — only needed during prefill (sl > 1)
    // During generation (sl = 1), each token only attends to the cache, no future tokens exist
    let mask : Tensor =
        if sl > 1 then
            let dev = h.device
            // Upper triangular matrix filled with −∞, diagonal=1 means the diagonal is also 0
            // m[i,j] = −∞ if j > i (token i cannot see token j if j is in the future)
            let m = torch.zeros(i64 sl, i64 sl, ...).fill_(Single.NegativeInfinity).triu(diagonal = 1)
            // hstack: prepend zeros for the cached positions (they are all in the past)
            // Final mask shape: [S_q, startPos + S_q]
            torch.hstack([| torch.zeros(i64 sl, i64 startPos, device = dev); m |]).type_as(h)
        else Unchecked.defaultof<Tensor>   // null in F# interop terms

    // Pass through all N transformer blocks sequentially
    let mutable h'' = h
    for i in 0..args.NLayers-1 do h'' <- layers.[i].forward(h'', startPos, f, mask)
    // Final RMSNorm + linear projection to vocabulary: [B, S, dim] → [B, S, vocab_size]
    output.forward(norm.forward(h''))
```

The mask construction deserves attention. During prefill, the new tokens form a `[S_q, S_q]` causal block. But these tokens also attend to cached tokens from previous calls — those are always in the past, so their mask entries are zero (no masking). `hstack` prepends a `[S_q, startPos]` block of zeros to the causal block, giving the full `[S_q, startPos + S_q]` mask.

---

## Tokenizer.fs: Tiktoken BPE

### What the Tokenizer Does

Before the model sees any text, every character must be converted to a token ID — an integer index into the model's vocabulary. Llama 3.x uses the same tokenizer as GPT-4: a **Byte Pair Encoding (BPE)** tokenizer with a vocabulary of 128,256 tokens, implemented in the style of OpenAI's Tiktoken.

BPE works bottom-up: start with individual bytes, then repeatedly merge the most frequent adjacent pair across all training text into a new token. The result is a vocabulary where common subwords, words, and even short phrases have single token IDs, while rare Unicode characters get split into their individual bytes (which are always in the vocabulary).

The tokenizer model file (`tokenizer.model`) stores this learned merge table: each line is a base64-encoded byte sequence and its rank (lower rank = merged earlier = more frequent):

```fsharp
do
    for line in File.ReadLines(tokenizerModelPath) do
        if not (String.IsNullOrWhiteSpace line) then
            let spaceIdx = line.LastIndexOf(' ')
            if spaceIdx > 0 then
                match Int32.TryParse(line.[spaceIdx + 1 ..]) with
                | true, rank ->
                    let b64   = line.[..spaceIdx - 1]
                    let bytes = Convert.FromBase64String(b64)
                    encoder.[b64] <- rank    // base64(bytes) → rank  (for encoding)
                    decoder.[rank] <- bytes  // rank → bytes          (for decoding)
                | _ -> ()
```

### Encoding: BPE Merging from Scratch

To encode a piece of text, first split it using the GPT-4 pre-tokenization regex (which handles contractions, punctuation, whitespace, and numbers), then apply BPE to each piece:

```fsharp
let encodePiece (bytes: byte[]) : int seq =
    // Start: each byte is its own "part"
    let parts = ResizeArray(bytes |> Array.map (fun b -> [| b |]))
    let mutable cont = parts.Count > 1
    while cont do
        // Find the adjacent pair with the lowest rank (= most frequent merge)
        let mutable minRank = Int32.MaxValue
        let mutable minIdx  = -1
        for i in 0 .. parts.Count - 2 do
            let merged = Array.append parts.[i] parts.[i + 1]
            let key    = Convert.ToBase64String(merged)
            match encoder.TryGetValue(key) with
            | true, rank when rank < minRank ->
                minRank <- rank
                minIdx  <- i
            | _ -> ()
        if minIdx = -1 then
            cont <- false   // no more merges possible
        else
            // Replace the two parts at minIdx with their merged form
            parts.[minIdx] <- Array.append parts.[minIdx] parts.[minIdx + 1]
            parts.RemoveAt(minIdx + 1)
    // Yield the rank of each final part
    seq {
        for part in parts do
            let key = Convert.ToBase64String(part)
            match encoder.TryGetValue(key) with
            | true, rank -> yield rank
            | _ -> ()
    }
```

The loop terminates when no adjacent pair is in the vocabulary — all remaining parts map directly to token IDs. Starting from individual bytes guarantees the tokenizer never fails on arbitrary UTF-8 input, since every byte value is in the vocabulary.

### Special Tokens and the Instruct Template

Llama 3 instruct models use a small set of special tokens that never appear in regular BPE encoding — they mark structural boundaries in the conversation format:

```fsharp
let specialTokens = dict [
    "<|begin_of_text|>",   128000   // marks the start of the entire sequence
    "<|end_of_text|>",     128001   // end of generation (base model)
    "<|start_header_id|>", 128006   // opens a role header (e.g., "system", "user")
    "<|end_header_id|>",   128007   // closes the role header
    "<|eot_id|>",          128009   // end of turn — generation stops here
]
```

During encoding, the text is first split on these special token patterns with a regex. Each plain-text segment is encoded with BPE, and special token IDs are inserted at the boundaries — they're never run through BPE (which would tokenize them as regular text and produce wrong IDs).

---

## Llama.fs: Loading and Generation

### Loading the Checkpoint

A PyTorch `.pth` file is a serialized Python dictionary mapping parameter names (like `"layers.0.attention.wq.weight"`) to tensors. `TorchSharp.PyBridge` reads this format and loads each tensor into the matching parameter of the F# model by matching names:

```fsharp
torch.set_default_dtype(torch.bfloat16)
let model = new Model.Transformer(modelArgs)
// load_py reads the .pth file and copies weights into the model's parameters by name
// strict = false: ignore .pth keys that don't match any parameter (e.g., rope.freqs)
model.load_py(location = weightPath, strict = false, loadedParameters = loadedParams) |> ignore
// Move all parameters and buffers to the target device (CPU or CUDA)
let model = model.``to``(device)
```

`torch.set_default_dtype(torch.bfloat16)` ensures that any tensor created without an explicit dtype (like the KV cache zeros) uses BFloat16, matching the checkpoint format. BFloat16 uses the same exponent range as Float32 but with fewer mantissa bits — it preserves dynamic range at half the memory cost.

### Top-p (Nucleus) Sampling

When `temperature > 0`, the model doesn't greedily pick the highest-probability token — it samples from a distribution. **Nucleus sampling** focuses that distribution: only the smallest set of tokens whose cumulative probability exceeds `topP` is considered, and the remaining tokens are zeroed out.

This prevents the model from occasionally sampling very unlikely tokens (which greedy avoids but temperature introduces), while still allowing diverse, non-repetitive generation:

```fsharp
member private _.SampleTopP (logits: Tensor) (topP: float32) : Tensor =
    // Sort tokens from highest to lowest probability
    let struct (probsSort, probsIndex) = torch.sort(logits, dim = -1L, descending = true)
    // cumsum[i] = sum of probabilities for tokens 0..i
    let cumsum  = torch.cumsum(probsSort, dim = -1L)
    // cumsum[i] - probsSort[i] = cumulative probability BEFORE token i
    // If this already exceeds topP, token i is outside the nucleus
    let mask    = torch.gt(cumsum - probsSort, torch.tensor(topP))
    // Zero out tokens outside the nucleus
    probsSort.masked_fill_(mask, 0.0f) |> ignore
    // Renormalize so the remaining probabilities sum to 1
    let probsSort' = probsSort / probsSort.sum(dim = -1L, keepdim = true)
    // Sample one token from the nucleus distribution
    let nextToken  = torch.multinomial(probsSort', num_samples = 1L)
    // Map back to original vocabulary indices (sort changed the order)
    torch.gather(probsIndex, dim = -1L, index = nextToken)
```

With `topP = 0.9`, typically a few dozen tokens form the nucleus. With `topP = 1.0`, all tokens are eligible (pure temperature sampling). With `temperature = 0` (not using this function), the highest-probability token is always chosen — deterministic greedy decoding.

### Autoregressive Generation: Prefill and Decode

Text generation happens in two distinct phases:

**Prefill** — The prompt tokens are processed all at once. The model runs a single forward pass with `sl = promptLength`, building the full KV cache for the prompt. This is efficient because all prompt tokens are processed in parallel.

**Decode** — One token is generated at a time. Each step: feed the single new token, run the forward pass (which now only processes `sl = 1` but attends to the entire cache), sample the next token, write it to the token buffer, repeat.

```fsharp
let mutable prevPos = 0
let mutable curPos  = minPrompt   // start of generation = end of shortest prompt

while curPos < totalLen && not stop do
    // On first iteration: prevPos=0, curPos=promptLen → processes full prompt (prefill)
    // On subsequent: prevPos=curPos-1, curPos → processes single new token (decode)
    let logits    = transformer.forward(tokens.narrow(1, i64 prevPos, i64(curPos - prevPos)), prevPos)
    // We only care about the last position's logits (the prediction for the next token)
    let lastLogit = logits.select(1, logits.shape.[1] - 1L)

    let nextToken =
        if temperature > 0f then
            // Apply temperature: higher T → flatter distribution → more random
            let probs = torch.softmax(lastLogit / temperature, dim = -1)
            this.SampleTopP probs topP
        else
            // Greedy: always pick the most probable token
            torch.argmax(lastLogit, dim = -1L)

    // Write the sampled token into the sequence buffer
    tokens.select(1, i64 curPos).copy_(nextToken) |> ignore

    // Track which sequences have finished (produced <|eot_id|>)
    // notMask: positions that were padding (not part of the original prompt)
    eosReached.bitwise_or_(torch.bitwise_and(notMask, torch.eq(nextToken, tokenizer.EosId))) |> ignore
    if eosReached.all().item<bool>() then stop <- true

    prevPos <- curPos
    curPos  <- curPos + 1
```

The `tokens` buffer is pre-allocated to `[batch, totalLen]` and filled with the pad token ID. As generation proceeds, the buffer is filled left-to-right. After the loop, the output strips everything after the first `<|eot_id|>` token.

---

## Program.fs: Interactive CLI with the Instruct Template

Llama 3 **instruct** models are fine-tuned to follow a very specific conversation format. If the input doesn't use this format exactly, the model won't behave as an assistant — it will treat the raw text as a continuation task and produce unexpected output.

The format for a single user turn looks like this:

```fsharp
let prompt =
    "<|start_header_id|>system<|end_header_id|>\n\n" +
    "You are a helpful assistant.<|eot_id|>" +
    "<|start_header_id|>user<|end_header_id|>\n\n" +
    input + "<|eot_id|>" +
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
```

Breaking it down token by token:

| Part | Meaning |
|---|---|
| `<\|start_header_id\|>system<\|end_header_id\|>\n\n` | Opens the system role header |
| `You are a helpful assistant.` | The system prompt — sets the assistant's persona |
| `<\|eot_id\|>` | End of the system turn |
| `<\|start_header_id\|>user<\|end_header_id\|>\n\n` | Opens the user role header |
| `input` | The user's actual message |
| `<\|eot_id\|>` | End of the user turn |
| `<\|start_header_id\|>assistant<\|end_header_id\|>\n\n` | Opens the assistant header — the model continues from here |

The generation call does *not* include a closing `<|eot_id|>` for the assistant — the model generates freely until it produces one itself. When the generation loop detects `<|eot_id|>` (token ID 128009), it stops and the response is complete.

---

## Running

1. Download [`Llama-3.2-1B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) weights and place them in a local folder (`consolidated.00.pth`, `params.json`, `tokenizer.model`).
2. Update `modelFolder` in `Program.fs`:

```fsharp
let modelFolder = @"C:\Models\Llama3.2-1B-Instruct"
```

3. Run:

```bash
dotnet run --project src -c Release
```

<video src="https://github.com/user-attachments/assets/4034b882-ebd3-4edd-952b-d9b8b1483716" controls width="100%"></video>

The first run downloads ~2 GB of NuGet packages (LibTorch with CUDA 12.1). Loading the checkpoint takes ~25 seconds on first run.

```
Using device: cuda
Loading checkpoint took 24521 ms
Model loaded. Type your prompt and press Enter (or 'exit' to quit).

You: _
```

---

## Sample Outputs

### Logic Reasoning

```
You: If A > B and B > C, which one is the greatest?
Assistant: If A > B and B > C, then A is the greatest.

You can think of it as a chain: A is greater than B, and B is greater than C,
so A is at the top of the chain and is therefore the greatest.
```

### Code Generation

```
You: Write a Python function that reverses a string.
Assistant: Here is a simple Python function that reverses a string:

def reverse_string(s):
    return s[::-1]

This function uses Python's slice notation with a step of -1 to reverse the string.
```

### Opinion

```
You: Do you like F#?
Assistant: F# is a functional-first programming language developed by Microsoft,
and it has a lot of features that many developers appreciate:

- Concise syntax: F# code is often more concise than equivalent C# code.
- Strong type system: helps catch errors at compile time.
- Functional programming: leads to more predictable and composable code.
```

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| [TorchSharp](https://github.com/dotnet/TorchSharp) | 0.101.5 | Tensor operations, neural network modules |
| `libtorch-cuda-12.1-win-x64` | 2.1.0.1 | Native LibTorch binaries with CUDA 12.1 |
| `TorchSharp.PyBridge` | 1.2.0 | Load PyTorch `.pth` checkpoints |
| [Microsoft.ML.Tokenizers](https://github.com/dotnet/machinelearning) | 0.21.1 | Tiktoken BPE tokenizer base |

---

## References

- **[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)** — Inspiration for running LLMs natively without Python.
- **[Microsoft.ML.GenAI.LLaMA](https://github.com/dotnet/machinelearning/tree/main/src/Microsoft.ML.GenAI.LLaMA)** — Official LLaMA in ML.NET, also on TorchSharp.
- **[hkproj/pytorch-llama](https://github.com/hkproj/pytorch-llama)** — Pedagogical reference for the Llama architecture.
- **[Coding LLaMA 2 from scratch in PyTorch](https://www.youtube.com/watch?v=oM4VmoabDAI)** — Umar Jamil
- **[But what is a GPT?](https://www.youtube.com/watch?v=eMlx5fFNoYc)** — 3Blue1Brown

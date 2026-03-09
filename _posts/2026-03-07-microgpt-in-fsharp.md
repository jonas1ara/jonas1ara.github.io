---
title: "MicroGPT in F# — Part 1: How Karpathy's GPT Works"
description: "How Andrej Karpathy's single-file GPT works: autograd, the transformer forward pass, and a Python vs F# comparison of every key piece."
Author: Jonas Lara
date: 2026-03-07 00:00:00 +0000
categories: [Functional Programming, F#, Machine Learning]
tags: [fsharp, gpt, transformers, dotnet, autograd, machine learning]
image:
  path: /assets/img/post/microgpt-in-fsharp/microgpt.png
  lqip: https://raw.githubusercontent.com/jonas1ara/jonas1ara.github.io/refs/heads/main/assets/img/post/microgpt-in-fsharp/microgpt.png
  alt: MicroGPT in F#
---

# MicroGPT in F#

[Andrej Karpathy's microgpt](https://karpathy.github.io/2026/02/12/microgpt/) is a ~200-line Python script that implements a GPT from scratch — no PyTorch, no NumPy, no frameworks of any kind. Just raw autograd, a transformer loop, and the essentials. It's a masterpiece for understanding how language models actually work.

[Martin Škuta](https://github.com/martinskuta/microgpt) then ported it to C# with serious .NET performance work: SIMD vectorization, iterative backward pass, zero-allocation hot paths, and loop unrolling.

This post is about [my F# port](https://github.com/jonas1ara/microgpt) that builds on that optimized C# foundation while embracing idiomatic F# design.

---

## How Karpathy's GPT Works

The Python original is deliberately minimal — every line teaches something. Let's go through the key pieces comparing Python and F# side by side.

### Autograd: Teaching a Number to Remember How It Was Born

The entire system is built on a single class — `Value` — which wraps a float and records the computation graph that produced it.

**Python:**

```python
import math

class Value:
    def __init__(self, data, _children=(), _local_grads=()):
        self.data = float(data)
        self.grad = 0.0
        self._children   = list(_children)
        self._local_grads = list(_local_grads)
```

**F#:**

```fsharp
type Value(data: float, children: Value[], localGrads: float[]) =
    let mutable _data = data
    let mutable _grad = 0.0

    member _.Data       with get() = _data and set(v) = _data <- v
    member _.Grad       with get() = _grad and set(v) = _grad <- v
    member _.Children   = children
    member _.LocalGrads = localGrads
```

The key difference: F# uses an array `float[]` for `localGrads` instead of a list. Arrays are contiguous in memory — critical for the SIMD optimization we'll see later.

---

### Operators: Where the Chain Rule Lives

Every arithmetic operation creates a new `Value` storing the result *and* the local partial derivatives of its inputs. This is the chain rule baked directly into the data structure.

**Python:**

```python
def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    return Value(self.data + other.data,
                 [self, other],
                 [1.0, 1.0])          # ∂(a+b)/∂a = 1, ∂(a+b)/∂b = 1

def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    return Value(self.data * other.data,
                 [self, other],
                 [other.data, self.data])  # ∂(ab)/∂a = b, ∂(ab)/∂b = a

def pow(self, exp):
    return Value(self.data ** exp,
                 [self],
                 [exp * self.data ** (exp - 1)])  # ∂(x^n)/∂x = n·x^(n-1)

def log(self):
    return Value(math.log(self.data),
                 [self],
                 [1.0 / self.data])    # ∂log(x)/∂x = 1/x

def exp(self):
    e = math.exp(self.data)
    return Value(e, [self], [e])       # ∂e^x/∂x = e^x

def relu(self):
    return Value(max(0, self.data),
                 [self],
                 [1.0 if self.data > 0 else 0.0])
```

**F#:**

```fsharp
static member (+)(a: Value, b: Value) =
    Value(a.Data + b.Data, [|a; b|], [|1.0; 1.0|])

static member (*)(a: Value, b: Value) =
    Value(a.Data * b.Data, [|a; b|], [|b.Data; a.Data|])

member this.Pow(other: float) =
    Value(Math.Pow(_data, other),
          [|this|],
          [|other * Math.Pow(_data, other - 1.0)|])

member this.Log() =
    Value(Math.Log(_data), [|this|], [|1.0 / _data|])

member this.Exp() =
    let e = Math.Exp _data
    Value(e, [|this|], [|e|])

member this.Relu() =
    Value(Math.Max(0.0, _data),
          [|this|],
          [|if _data > 0.0 then 1.0 else 0.0|])
```

Both versions encode the same calculus. The F# version uses static operator overloading instead of dunder methods, and `[|...|]` (array literals) instead of Python lists.

---

### Backward Pass: Propagating Gradients

Once the forward pass completes and produces a loss, `backward()` walks the computation graph in **reverse topological order**, multiplying local gradients by upstream gradients (the chain rule):

```
loss.grad = 1.0
for each node v (in reverse topo order):
    for each child c of v:
        c.grad += v.grad * local_grad(v → c)
```

**Python (recursive DFS):**

```python
def backward(self):
    topo    = []
    visited = set()

    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._children:
                build_topo(child)      # ← recursive call
            topo.append(v)

    build_topo(self)
    self.grad = 1.0

    for v in reversed(topo):
        if v.grad != 0.0:
            for child, lg in zip(v._children, v._local_grads):
                child.grad += lg * v.grad
```

**F# (iterative DFS with explicit stack):**

```fsharp
member this.Backward
    (topo: ResizeArray<Value>, visited: HashSet<Value>,
     stack: Stack<struct (Value * int)>) =

    stack.Push(struct (this, 0))

    while stack.Count > 0 do
        let struct (current, childIndex) = stack.Pop()
        let ch = current.Children

        if ch.Length > 0 && childIndex < ch.Length then
            stack.Push(struct (current, childIndex + 1))
            let child = ch.[childIndex]
            if visited.Add child then
                stack.Push(struct (child, 0))
        else
            topo.Add current

    this.Grad <- 1.0

    for topoIdx = topo.Count - 1 downto 0 do
        let v     = topo.[topoIdx]
        let vGrad = v.Grad
        if vGrad <> 0.0 then
            let ch  = v.Children
            let lg  = v.LocalGrads
            let len = ch.Length
            match len with
            | 1 -> ch.[0].Grad <- ch.[0].Grad + lg.[0] * vGrad
            | 2 -> ch.[0].Grad <- ch.[0].Grad + lg.[0] * vGrad
                   ch.[1].Grad <- ch.[1].Grad + lg.[1] * vGrad
            | _ -> for i in 0 .. len - 1 do
                       ch.[i].Grad <- ch.[i].Grad + lg.[i] * vGrad
```

Python's version is elegant but **recursive** — for a deep computation graph (many layers × long sequences), it will hit the call stack limit. The F# version simulates the same DFS with an explicit `Stack`, avoiding that limit entirely. More on this in the optimizations section.

---

### Utility Functions: softmax and rmsNorm

These are the building blocks used inside the transformer.

**Python:**

```python
def softmax(logits):
    max_val = max(v.data for v in logits)          # numerical stability
    exps    = [(v - max_val).exp() for v in logits]
    total   = sum(exps, Value(0.0))
    return [e / total for e in exps]

def rms_norm(x):
    sum_sq = sum((xi * xi for xi in x), Value(0.0))
    ms     = sum_sq / len(x)
    scale  = (ms + 1e-5).pow(-0.5)
    return [xi * scale for xi in x]
```

**F#:**

```fsharp
let softmax (logits: ResizeArray<Value>) =
    let maxVal = logits |> Seq.map (fun v -> v.Data) |> Seq.max
    let exps   = ResizeArray(logits |> Seq.map (fun v -> (v - maxVal).Exp()))
    let total  = exps |> Seq.fold (fun acc e -> acc + e) (Value 0.0)
    ResizeArray(exps |> Seq.map (fun e -> e / total))

let rmsNorm (x: ResizeArray<Value>) =
    let sumSq = x |> Seq.fold (fun acc xi -> acc + xi * xi) (Value 0.0)
    let ms    = sumSq / float x.Count
    let scale = (ms + 1e-5).Pow(-0.5)
    ResizeArray(x |> Seq.map (fun xi -> xi * scale))
```

The logic is identical. The F# version reads left-to-right with `|>` pipelines; Python uses list comprehensions.

---

### The Transformer: gpt()

The `gpt` function is the heart of the model. It takes one token and its position, and returns a vector of logits — one score per vocabulary token, representing "how likely is this token to come next?". Let's go through each step.

---

#### Step 1 — Token + Positional Embeddings

Each token id maps to a learned vector (what the token *means*). Each position id maps to another learned vector (where the token *is* in the sequence). They are added together to form the initial representation `x`.

**Python:**
```python
tok_emb = state["wte"][token_id]   # shape: [n_embd]
pos_emb = state["wpe"][pos_id]     # shape: [n_embd]
x = [t + p for t, p in zip(tok_emb, pos_emb)]
x = rms_norm(x)
```

**F#:**
```fsharp
let tokEmb = stateDict.["wte"].[tokenId]   // shape: [nEmbd]
let posEmb = stateDict.["wpe"].[posId]     // shape: [nEmbd]
let mutable x =
    ResizeArray(Array.init nEmbd (fun i -> tokEmb.[i] + posEmb.[i]))
x <- rmsNorm x
```

After adding both embeddings, RMS normalization scales `x` so its values don't explode as they flow through the network. Think of it as a reset before each layer — the model learns *what* to emphasize, not just raw magnitudes.

---

#### Step 2 — Q, K, V Projections

For each transformer layer, we first save the current `x` as a residual (to add back later), then project `x` into three separate spaces: **Query**, **Key**, and **Value**. These are just learned linear transformations — three weight matrices multiplied by `x`.

**Python:**
```python
x_residual = list(x)
x = rms_norm(x)

q = linear(x, state[f"layer{li}.attn_wq"])   # What am I looking for?
k = linear(x, state[f"layer{li}.attn_wk"])   # What do I contain?
v = linear(x, state[f"layer{li}.attn_wv"])   # What will I contribute?

keys[li].append(k)      # KV cache: remember all past keys
values[li].append(v)    # KV cache: remember all past values
```

**F#:**
```fsharp
let xResidual = ResizeArray x
x <- rmsNorm x

let q = linear x stateDict.[sprintf "layer%d.attn_wq" li]
let k = linear x stateDict.[sprintf "layer%d.attn_wk" li]
let v = linear x stateDict.[sprintf "layer%d.attn_wv" li]

keys.[li].Add k      // KV cache
values.[li].Add v
```

The **KV cache** (`keys[li]`, `values[li]`) accumulates all past keys and values. At position `t`, the model can look at every previous token — this is the "context window" in action.

`linear` is just a matrix-vector product: for each row `w` in the weight matrix, compute `dot(w, x)` and collect the results. The result has the same dimension as `x` (`n_embd`).

---

#### Step 3 — Multi-Head Scaled Dot-Product Attention

This is the most important part. Instead of one attention mechanism over the full `n_embd` dimension, we split into `n_head` independent heads, each operating on a `head_dim = n_embd / n_head` slice. Each head learns to attend to different aspects of the context.

For each head `h`:

1. Extract the `head_dim`-sized slice of Q from position `h * head_dim`
2. For every past token `t`, compute a dot product between current Q and past K — this is the "how relevant is token t to my current query?" score
3. Divide by `√head_dim` (scaled dot-product) — without this scaling, dot products grow with `head_dim` because they are a sum of `head_dim` multiplications. Large values push softmax into saturation regions where gradients vanish to near zero, making training unstable
4. Softmax over all scores to get attention weights (they sum to 1)
5. Compute the weighted sum of all past V vectors — the output is "what I should attend to"

**Python:**
```python
x_attn = []
for h in range(n_head):
    hs  = h * head_dim
    q_h = q[hs : hs + head_dim]   # query for this head
    T   = len(keys[li])            # number of past tokens

    # Score: how much does current token attend to each past token?
    attn_logits = []
    for t in range(T):
        k_h = keys[li][t][hs : hs + head_dim]
        dot = sum(qi * ki for qi, ki in zip(q_h, k_h))
        attn_logits.append(dot / math.sqrt(head_dim))   # scale

    attn_weights = softmax(attn_logits)   # probabilities over past tokens

    # Weighted sum of values
    head_out = [Value(0.0)] * head_dim
    for t in range(T):
        v_h = values[li][t][hs : hs + head_dim]
        w   = attn_weights[t]
        for j in range(head_dim):
            head_out[j] = head_out[j] + w * v_h[j]

    x_attn.extend(head_out)
```

**F#:**
```fsharp
let xAttn = ResizeArray<Value>()
for h in 0 .. nHead - 1 do
    let hs = h * headDim
    let qH = q.GetRange(hs, headDim)   // query slice for this head
    let T  = keys.[li].Count           // number of past tokens

    // Score each past token against current query
    let attnLogits = ResizeArray<Value>()
    for t in 0 .. T - 1 do
        let kH = keys.[li].[t].GetRange(hs, headDim)
        let mutable dot = Value 0.0
        for j in 0 .. headDim - 1 do
            dot <- dot + qH.[j] * kH.[j]
        attnLogits.Add(dot / Math.Sqrt(float headDim))   // scale

    let attnWeights = softmax attnLogits   // probabilities

    // Weighted sum of values
    let headOut = ResizeArray(Array.init headDim (fun _ -> Value 0.0))
    for t in 0 .. T - 1 do
        let vH = values.[li].[t].GetRange(hs, headDim)
        let w  = attnWeights.[t]
        for j in 0 .. headDim - 1 do
            headOut.[j] <- headOut.[j] + w * vH.[j]

    xAttn.AddRange headOut
```

After the loop, `xAttn` contains the concatenated output of all heads — shape `[n_embd]`.

---

#### Step 4 — Output Projection + First Residual

The concatenated attention output is projected back to `n_embd` via `attn_wo`, then added to the residual saved in Step 2. This residual connection is crucial — it lets gradients flow directly back through the network during training, preventing vanishing gradients.

**Python:**
```python
x = linear(x_attn, state[f"layer{li}.attn_wo"])   # mix heads back together
x = [xi + ri for xi, ri in zip(x, x_residual)]    # + residual
```

**F#:**
```fsharp
x <- linear xAttn stateDict.[sprintf "layer%d.attn_wo" li]
for i in 0 .. nEmbd - 1 do
    x.[i] <- x.[i] + xResidual.[i]   // + residual
```

---

#### Step 5 — MLP Block + Second Residual

Every transformer layer also has a feed-forward network (MLP) applied position-wise. It expands the dimension by 4× (`mlp_fc1`), applies a non-linearity, then compresses back (`mlp_fc2`). Another residual connection wraps it.

> **ReLU, not GELU.** Modern transformers (GPT-2, GPT-3, LLaMA) use GELU as the MLP activation — it's smooth and has non-zero gradients for slightly negative inputs, which improves training dynamics. microgpt deliberately uses **ReLU** (`max(0, x)`) for simplicity: it's dead-simple to implement in scalar autograd (gradient is exactly 0 or 1), and it's enough for a character-level toy model. If you wanted to extend this to a larger model, swapping ReLU for GELU would be one of the first changes to make.

**Python:**
```python
x_residual2 = list(x)
x = rms_norm(x)
x = linear(x, state[f"layer{li}.mlp_fc1"])   # expand: n_embd → 4*n_embd
x = [xi.relu() for xi in x]                  # ReLU (not GELU — intentional)
x = linear(x, state[f"layer{li}.mlp_fc2"])   # compress: 4*n_embd → n_embd
x = [xi + ri for xi, ri in zip(x, x_residual2)]   # + residual
```

**F#:**
```fsharp
let xResidual2 = ResizeArray x
x <- rmsNorm x
x <- linear x stateDict.[sprintf "layer%d.mlp_fc1" li]   // expand
x <- ResizeArray(x |> Seq.map (fun xi -> xi.Relu()))      // ReLU (not GELU — intentional)
x <- linear x stateDict.[sprintf "layer%d.mlp_fc2" li]   // compress
for i in 0 .. nEmbd - 1 do
    x.[i] <- x.[i] + xResidual2.[i]   // + residual
```

Why 4×? This is a heuristic from the original "Attention is All You Need" paper — empirically it gives enough capacity for the MLP to learn useful transformations without being too expensive.

---

#### Step 6 — Language Model Head

After all layers, `x` is a rich vector representation of the current token in its context. The final linear projection maps it to vocabulary size — one score per possible next token.

**Python:**
```python
return linear(x, state["lm_head"])   # shape: [vocab_size]
```

**F#:**
```fsharp
linear x stateDict.["lm_head"]   // shape: [vocabSize]
```

These raw scores are the **logits**. Pass them through softmax to get probabilities, then pick the highest-probability token (or sample from the distribution) to generate the next character.

---

Putting it all together, one forward pass looks like this:

```
token_id, pos_id
    │
    ▼
[tok_emb + pos_emb]  →  rmsNorm  →  x        (shape: n_embd)
    │
    ▼  (repeated n_layer times)
  save x_residual
  rmsNorm(x)
  Q, K, V = linear(x, Wq), linear(x, Wk), linear(x, Wv)
  append K, V to KV cache
  for each head h:
      score each past K against current Q  →  softmax  →  weights
      weighted sum of past V               →  head_out
  concat all heads  →  linear(attn_wo)  +  x_residual
  save x_residual2
  rmsNorm  →  linear(fc1)  →  relu  →  linear(fc2)  +  x_residual2
    │
    ▼
  linear(lm_head)  →  logits   (shape: vocab_size)
```

---

### Python vs F# at a Glance

| Component | Python (microgpt) | F# (this impl) | Advantage in F# |
|---|---|---|---|
| **Autograd** | `Value` class with Python lists | `Value` with flat `float[]` arrays | Better memory locality, SIMD-ready |
| **Backward pass** | Recursive DFS | Iterative with `Stack<struct>` | No stack overflow, zero heap alloc per node |
| **Operators** | Dunder methods (`__mul__`, `__add__`) | Static members + operator overloading | More idiomatic, composable |
| **KV cache** | Python lists with `.append()` | `ResizeArray` (controlled mutability) | Explicit mutation intent, same O(1) amortized |
| **Activation** | `relu` (scalar) | `Relu()` (scalar) | Identical — ReLU intentional for simplicity |
| **Dot product** | `sum(a * b for a, b in zip(...))` | SIMD `Vector<float>` loop | 4× throughput on AVX2 |
| **Data flow** | Imperative rebinding | `|>` pipelines + explicit `mutable` | Mutation visible and localized |
| **Composition** | Lambdas / loops | `Seq.map`, `Seq.fold`, `Array.init` | Reads as a transformation pipeline |

---

### Training Loop

**Python:**

```python
for step in range(num_steps):
    doc    = docs[step % len(docs)]
    tokens = [bos] + [encode(c) for c in doc] + [bos]
    n      = min(block_size, len(tokens) - 1)

    keys   = [[] for _ in range(n_layer)]
    values = [[] for _ in range(n_layer)]

    losses = []
    for pos_id in range(n):
        token_id  = tokens[pos_id]
        target_id = tokens[pos_id + 1]
        logits    = gpt(token_id, pos_id, keys, values)
        probs     = softmax(logits)
        losses.append(-probs[target_id].log())

    loss = sum(losses, Value(0.0)) * (1.0 / n)

    for p in params:
        p.grad = 0.0
    loss.backward()

    # Adam update
    lr_t = learning_rate * (1.0 - step / num_steps)
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (math.sqrt(v_hat) + eps)
```

**F#:**

```fsharp
for step in 0 .. numSteps - 1 do
    let doc    = docs.[step % docs.Count]
    let tokens = ResizeArray<int>()
    tokens.Add bos
    tokens.AddRange(doc |> Seq.map encode)
    tokens.Add bos
    let n = min blockSize (tokens.Count - 1)

    let keys   = Array.init nLayer (fun _ -> ResizeArray<ResizeArray<Value>>())
    let values = Array.init nLayer (fun _ -> ResizeArray<ResizeArray<Value>>())

    let losses = ResizeArray<Value>()
    for posId in 0 .. n - 1 do
        let tokenId  = tokens.[posId]
        let targetId = tokens.[posId + 1]
        let logits   = gpt tokenId posId keys values
        let probs    = softmax logits
        losses.Add(-probs.[targetId].Log())

    let mutable loss = Value 0.0
    for l in losses do loss <- loss + l
    loss <- loss * (1.0 / float n)

    for p in paramsList do p.Grad <- 0.0
    topo.Clear()
    visited.Clear()
    stack.Clear()
    loss.Backward(topo, visited, stack)

    // Adam update with linear LR decay
    let lrT = learningRate * (1.0 - float step / float numSteps)
    for i in 0 .. paramsList.Length - 1 do
        let p = paramsList.[i]
        mArr.[i] <- beta1 * mArr.[i] + (1.0 - beta1) * p.Grad
        vArr.[i] <- beta2 * vArr.[i] + (1.0 - beta2) * (p.Grad ** 2.0)
        let mHat = mArr.[i] / (1.0 - beta1 ** float (step + 1))
        let vHat = vArr.[i] / (1.0 - beta2 ** float (step + 1))
        p.Data <- p.Data - lrT * mHat / (Math.Sqrt vHat + epsAdam)
```

One notable difference: in F# the `topo`, `visited`, and `stack` collections are **allocated once outside the loop** and `.Clear()`-ed each step. Python creates fresh objects every iteration. This is one of the zero-allocation optimizations — more on that next.

---

*Continue in [Part 2](/posts/microgpt-in-fsharp-part-2): .NET performance optimizations, F# vs C# idioms, training runs, and what it would take to scale.*

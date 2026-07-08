# llama.cpp Internal Principles

A beginner-friendly walkthrough of what llama.cpp does internally, from zero knowledge up to how the code is organized.

---

## 1. What is "inference"?

An LLM (Large Language Model) is essentially a giant mathematical function trained on text. It learned patterns like:

> "After `Hello my name is`, the next word is often a name."

- **Training** = teaching the model (expensive, done once by model creators).
- **Inference** = using the trained model to produce output (what llama.cpp does).

When you type a prompt and get a reply, inference is:

1. Turn your text into numbers (**tokens**)
2. Run those numbers through the model
3. Get a probability distribution over the next token
4. Pick one token
5. Append it and repeat until done

That loop is the heart of everything in this project.

---

## 2. What llama.cpp is

From the README:

> *"The main goal of llama.cpp is to enable LLM inference with minimal setup and state-of-the-art performance on a wide range of hardware."*

In plain terms:

- Load a pretrained model from disk
- Run it efficiently on **CPU, GPU, Apple Silicon**, etc.
- Generate text token by token
- Expose that through CLI tools, a library (`libllama`), and an HTTP server

It is **not** a training framework. It is a **runtime engine** for running models.

---

## 3. The big picture: layers of the stack

```
  You / Your App
  +------------------+  +------------------+  +------------------+
  |    llama-cli     |  |   llama-server   |  | app via libllama |
  +--------+---------+  +--------+---------+  +--------+---------+
           \___________________|___________________/
                               |
                               v
                    llama.cpp (model logic)
           +------------------------------------------+
           | Tokenizer / vocab                        |
           |        |                                 |
           |        v                                 |
           | Transformer graph builder                |
           |        |                                 |
           |        v                                 |
           | KV cache                                 |
           |        |                                 |
           |        v                                 |
           | Sampler (pick next token)                |
           +--------+---------------------------------+
                    |
                    v
                    GGML (tensor math engine)
           +------------------------------------------+
           | Tensors & operations                     |
           |        |                                 |
           |        v                                 |
           | Computation graph                        |
           |        |                                 |
           |        v                                 |
           | Backends: CPU, CUDA, Metal, Vulkan...    |
           +------------------------------------------+

  model.gguf (on disk)  ----loads into---->  Tokenizer / vocab
```

| Layer | Role |
|-------|------|
| **GGUF file** | The model weights + metadata on disk |
| **GGML** | Low-level tensor math (matrix multiply, attention, etc.) |
| **llama.cpp** | LLM-specific logic: architecture, tokenizer, KV cache, sampling |
| **tools/** | User-facing programs: CLI, server, quantize, bench, UI |

---

## 4. The model file: GGUF

Models are stored as `.gguf` files (GGML Unified Format).

A GGUF file contains:

- **Weights** - billions of numbers that define the model (often quantized to 4-bit, 8-bit, etc. to save memory)
- **Metadata** - architecture type, context length, tokenizer info
- **Vocabulary** - how text maps to token IDs

You often download or convert models from Hugging Face into GGUF. The `convert_hf_to_gguf.py` script in the repo does that conversion.

**Quantization** (e.g. Q4_K_M) means storing weights in fewer bits. You trade a small quality loss for much less RAM and faster inference. The `tools/quantize/` tool handles this.

---

## 5. What is a Transformer? (the model architecture)

Most models llama.cpp supports are **Transformers**. Conceptually, each layer does:

```
Token embeddings
        |
        v
Self-Attention  (look at previous tokens)
        |
        v
Feed-Forward Network  (transform each position)
        |
        v
Next layer  (repeat)  or  logits  (final layer)
```

For each token position, the model:

1. **Embeds** the token ID into a vector
2. **Attention** - "Which previous tokens should I pay attention to?"
3. **FFN** - Nonlinear transformation
4. Repeat for N layers (e.g. 32 layers for a 7B model)
5. **Output head** - Produces scores (logits) for every possible next token

The code that wires this up lives mainly in `src/llama-graph.cpp` (builds the computation graph) and `src/llama-arch.cpp` (model-specific layouts).

---

## 6. The core inference loop

The simplest end-to-end example is `examples/simple/simple.cpp`. The flow is:

### Step A: Load model and create context

```cpp
llama_model * model = llama_model_load_from_file("model.gguf", model_params);
llama_context * ctx = llama_init_from_model(model, ctx_params);
```

- `llama_model` = read-only weights + architecture
- `llama_context` = runtime state (KV cache, buffers, backends)

### Step B: Tokenize the prompt

```cpp
llama_tokenize(vocab, "Hello my name is", ...);
// -> [15496, 590, 1638, 318]  (example token IDs)
```

Text becomes a sequence of integers. Different models use different tokenizers (BPE, SentencePiece, etc.).

### Step C: Prefill (process the prompt)

```cpp
llama_batch batch = llama_batch_get_one(prompt_tokens, n_prompt);
llama_decode(ctx, batch);  // process ALL prompt tokens at once
```

This is the **prefill** phase: run the full prompt through the model in one (or a few) batches. It is often slower per token but happens once.

### Step D: Decode loop (generate one token at a time)

```cpp
for (...) {
    llama_decode(ctx, batch);           // run model forward pass
    new_token_id = llama_sampler_sample(smpl, ctx, -1);  // pick next token
    // print token, append to batch, repeat
}
```

This is the **decode** (or **generation**) phase: one new token per `llama_decode` call. Speed here is usually measured in **tokens/second**.

### Step E: Sampling

The model outputs **logits** (raw scores for every token in the vocabulary). The **sampler** turns that into one chosen token:

- **Greedy** - always pick highest score
- **Temperature** - add randomness
- **Top-k / Top-p** - limit choices to likely tokens

Samplers live in the `llama_sampler` API in `include/llama.h`.

---

## 7. KV Cache - the key optimization

Without a KV cache, each new token would require reprocessing the entire history. That would be O(n^2) and far too slow.

The **KV cache** stores intermediate **Key** and **Value** tensors from attention for all previous tokens. When generating token 100, the model only computes attention for the new token and reuses cached K/V for tokens 1-99.

```
Token 1  -->  K,V stored  --+
                             \
Token 2  -->  K,V stored  ---+-->  Attention (token 3, new)
                             /         uses KV1 + KV2 + new K,V
Token 3  -->  compute new K,V --+
     (new)
```

Implementation: `src/llama-kv-cache.h` and related files. Context size (`n_ctx`) is how many tokens of history the cache can hold.

---

## 8. GGML: the math engine underneath

**GGML** is the tensor library llama.cpp is built on. It provides:

- **Tensors** - multidimensional arrays (weights, activations)
- **Operations** - `MUL_MAT`, `ROPE`, `RMS_NORM`, `FLASH_ATTN_EXT`, etc. (see `docs/ops.md`)
- **Computation graph** - operations chained together; executed lazily
- **Backends** - where math actually runs:

| Backend | Hardware |
|---------|----------|
| `ggml-cpu` | CPU (AVX, NEON, AMX...) |
| `ggml-cuda` | NVIDIA GPU |
| `ggml-metal` | Apple Silicon GPU |
| `ggml-vulkan` | Cross-platform GPU |
| `ggml-sycl` | Intel GPU |
| ... | Many more |

`n_gpu_layers` controls how many transformer layers are offloaded to GPU; the rest stay on CPU. That enables **hybrid inference** for models larger than VRAM.

---

## 9. Codebase map (mental model)

| Directory | Purpose |
|-----------|---------|
| `include/llama.h` | Public C API |
| `src/llama-*.cpp` | Core: model loading, context, graph, KV cache, sampling |
| `ggml/` | Tensor library + hardware backends |
| `common/` | Shared utilities (arg parsing, chat templates, logging) |
| `tools/cli/` | Command-line chat/completion |
| `tools/server/` | HTTP API server (OpenAI-compatible) |
| `tools/quantize/` | Model quantization |
| `tools/ui/` | Web UI for the server |
| `examples/simple/` | Minimal inference example |
| `conversion/` | Scripts to convert HuggingFace -> GGUF |

---

## 10. Server mode (how apps use it)

`llama-server` wraps the same core loop in HTTP:

```
Client                    llama-server              llama_context
  |                            |                         |
  |-- POST /v1/chat/completions ------------------------>|
  |                            |-- tokenize + prefill -->|
  |                            |                         |
  |                            |     +----------------+  |
  |                            |     | each new token |  |
  |                            |     | decode+sample  |  |
  |                            |     +----------------+  |
  |<-- SSE stream chunk -------|                         |
  |     (repeat until done)    |                         |
```

From `tools/server/README-dev.md`:

- **`server_context`** - holds the main `llama_context`
- **`server_slot`** - one parallel conversation/request
- Multiple slots = handle several users at once (batched inference)

---

## 11. End-to-end summary

```
User prompt: "Hello"
        |
        v
    Tokenizer
        |
        v
Token IDs: [15496]
        |
        v
llama_decode (prefill)
        |
        v
   KV cache filled
        |
        v
Logits for next token
        |
        v
Sampler picks token
        |
        v
Decode to text: " there"
        |
        v
     End? ----no----> Append token, llama_decode again ----+
        |                                                  |
       yes                                                 |
        |                                                  |
        v                                                  |
  Full response  <-----------------------------------------+
```

**In one sentence:** llama.cpp loads a quantized transformer model from a GGUF file, runs it through optimized tensor math (GGML) on your hardware, uses a KV cache to avoid recomputing history, and repeatedly samples the next token until generation stops.

---

## 12. Good places to read next

If you want to go deeper in the repo:

1. **`examples/simple/simple.cpp`** - ~220 lines, full inference loop
2. **`include/llama.h`** - public API surface
3. **`docs/ops.md`** - what tensor ops exist and which backends support them
4. **`tools/server/README-dev.md`** - how the HTTP server maps to `llama_context`
5. **`docs/development/HOWTO-add-model.md`** - how new model architectures are added

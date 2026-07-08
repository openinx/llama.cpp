# llama.cpp Book

A short guide to understanding how [llama.cpp](https://github.com/ggml-org/llama.cpp) runs large language models locally.

This book is managed with [GitBook](https://www.gitbook.com/) and synced from the `book/` directory in the llama.cpp repository.

## Who is this for?

- Developers new to AI model inference
- Contributors who want a mental model before reading the C/C++ source
- Anyone curious about what happens between a prompt and a generated reply

## What you will learn

1. **Internal principles** - GGUF, GGML, transformers, KV cache, sampling, and how the codebase is organized
2. **Choosing the next token** - Logits, softmax, and greedy vs sampling, explained for beginners
3. **Inference math** - The formulas behind next-token prediction (attention, RoPE, SwiGLU, softmax sampling)

## How to read

Chapters are ordered from high-level concepts to mathematical detail. Start with [Internal Principles](01-llama-cpp-internal-principles.md), then [How Does an LLM Choose the Next Token?](03-how-does-an-llm-choose-the-next-token.md), then [Next-Token Inference Math](02-next-token-inference-math.md).

## GitBook setup

To publish this book on GitBook:

1. Create a space on [gitbook.com](https://www.gitbook.com/)
2. Enable **Git Sync** and connect this repository
3. Set the **Project directory** to `book`
4. GitBook reads `.gitbook.yaml`, `README.md`, and `SUMMARY.md` from that folder

Local edits: change markdown files under `book/`, commit, and push. GitBook syncs automatically.

## Local preview server

Math formulas use the [KaTeX](https://katex.org/) plugin (`book.json`). To preview locally:

```sh
cd book
npm install
npx gitbook install
npm run serve
```

Open http://localhost:4000 in your browser.

To build static HTML:

```sh
npm run build
```

Output goes to `book/_book/`.

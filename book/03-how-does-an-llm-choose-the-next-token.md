# How Does an LLM Choose the Next Token?

This chapter explains, step by step, how a large language model picks the next word (token). You will learn what a **logit** is, how **softmax** turns scores into probabilities, and why the model does not always pick the most likely token.

If you already read [llama.cpp Internal Principles](01-llama-cpp-internal-principles.md), you saw the high-level loop: read context, get scores, pick a token, repeat. This chapter zooms in on the middle part - scores and probabilities - without heavy math. For formulas tied to the llama.cpp source, continue to [Next-Token Inference Math](02-next-token-inference-math.md).

---

## 1. One token at a time

A language model does **not** write a full sentence in one shot. It works like autocomplete, one piece at a time.

**Prompt:**

> Tonight I want to eat

**Question the model asks itself:**

> What word should come next?

Some candidates:

| Token | Rough meaning |
|-------|---------------|
| hotpot | food |
| noodles | food |
| car | vehicle |
| because | connector word |

The model first gives **every possible next token a score**. Only after that does it choose one token, append it to the text, and ask the same question again.

```
Read context ("Tonight I want to eat")
        |
        v
Score every possible next token  -->  logits
        |
        v
Turn scores into probabilities  -->  softmax
        |
        v
Pick one token  -->  greedy or sample
        |
        v
Append token and repeat
```

That loop is the whole generation process.

---

## 2. Scores first, probabilities later

Think of the model like a teacher grading a multiple-choice question with thousands of options.

The teacher does **not** start by saying "the answer is hotpot."  
First, they write a **raw score** next to each option:

| Token | Raw score |
|-------|----------:|
| hotpot | 10 |
| noodles | 7.5 |
| car | 0 |
| because | 0 |

In ML, these raw scores are called **logits** (pronounced "LOH-jits").

**Key idea:** a higher logit means the model likes that token more as the next word.

Logits are **not** percentages yet. They:

- can be negative
- can be larger than 1
- do **not** add up to 100%

So logits answer: "How much does the model prefer each option?"  
They do **not** yet answer: "What is the chance of each option?"

---

## 3. Where do logits come from? (simple picture)

Inside a real model, the math is huge. But the **idea** is simple: compare "what the context looks like" with "what each token looks like."

### Step 1: Turn context into a vector

After reading:

> Tonight I want to eat

the model builds a compact summary - a list of numbers called a **context vector**.

Toy example with only 3 numbers:

```text
context = [5, 0, 0]
```

Imagine what each slot means:

| Slot | Meaning in this toy example |
|------|----------------------------|
| 1st | "food-ness" |
| 2nd | "vehicle-ness" |
| 3rd | "connector-word-ness" |

So `[5, 0, 0]` roughly means:

> This sentence strongly points toward food.

A real model uses thousands of dimensions, not 3. The idea is the same: the context becomes one vector of numbers.

### Step 2: Each token also has a vector

During training, the model learned a vector for every token:

| Token | Token vector |
|-------|-------------|
| hotpot | `[2, 0, 0]` |
| noodles | `[1.5, 0, 0]` |
| car | `[0, 2, 0]` |
| because | `[0, 0, 2]` |

Notice:

- food words point along the 1st dimension
- `car` points along the 2nd
- `because` points along the 3rd

### Step 3: Compare with a dot product

The model compares context and token vectors using a **dot product**: multiply matching positions and add the results.

If context is `h = [5, 0, 0]`:

**hotpot** `[2, 0, 0]`:

```text
5 x 2 + 0 x 0 + 0 x 0 = 10
```

**noodles** `[1.5, 0, 0]`:

```text
5 x 1.5 + 0 x 0 + 0 x 0 = 7.5
```

**car** `[0, 2, 0]`:

```text
5 x 0 + 0 x 2 + 0 x 0 = 0
```

**because** `[0, 0, 2]`:

```text
5 x 0 + 0 x 0 + 0 x 2 = 0
```

Final logits:

| Token | Logit |
|-------|------:|
| hotpot | 10 |
| noodles | 7.5 |
| car | 0 |
| because | 0 |

**Intuition:** when context and token point in similar directions, the dot product is large. That is why `hotpot` and `noodles` score high here, while `car` and `because` do not.

In a real LLM, a deep **transformer** builds the context vector and an **output head** computes logits over the full vocabulary (often 50,000+ tokens). The toy dot-product story captures the core idea: *match context to candidate tokens*.

---

## 4. Why we need softmax

Suppose logits are:

| Token | Logit |
|-------|------:|
| hotpot | 3 |
| noodles | 2 |
| car | 0 |

You cannot treat these as probabilities:

```text
3 + 2 + 0 = 5   (not 1.0)
```

We need a conversion step that:

1. makes every value positive
2. scales them so they sum to 1 (100%)

That step is **softmax**.

---

## 5. Softmax in plain English

For each token:

```text
probability = e^(logit) / (sum of e^(all logits))
```

In symbols:

```text
p_i = e^(z_i) / sum of e^(z_j)
```

| Symbol | Meaning |
|--------|---------|
| `z_i` | logit of token `i` |
| `e^(z_i)` | exponent of that logit (always positive) |
| `p_i` | final probability of token `i` |

If exponent math feels unfamiliar, treat `e^x` as a button that:

- turns any score into a positive number
- makes bigger scores grow much faster than smaller ones

---

## 6. Worked example: logits to probabilities

Logits:

| Token | Logit |
|-------|------:|
| hotpot | 3 |
| noodles | 2 |
| car | 0 |

### Step 1: Apply `e^x`

```text
e^3 = 20.09   (about)
e^2 =  7.39   (about)
e^0 =  1.00
```

| Token | Logit | e^logit |
|-------|------:|--------:|
| hotpot | 3 | 20.09 |
| noodles | 2 | 7.39 |
| car | 0 | 1.00 |

### Step 2: Add them

```text
20.09 + 7.39 + 1.00 = 28.48
```

### Step 3: Divide each by the total

**hotpot:** `20.09 / 28.48 = 0.705` -> **70.5%**  
**noodles:** `7.39 / 28.48 = 0.260` -> **26.0%**  
**car:** `1.00 / 28.48 = 0.035` -> **3.5%**

| Token | Probability |
|-------|------------:|
| hotpot | 70.5% |
| noodles | 26.0% |
| car | 3.5% |

Now they add to 100%. These are real probabilities you can sample from.

**Check your intuition:** `hotpot` had the highest logit, so it got the highest probability. But `noodles` still has a real chance (26%), and `car` is unlikely but not impossible.

---

## 7. Why use `e^x` specifically?

Four practical reasons:

### 1) Always positive

Probabilities cannot be negative. Even for a negative logit:

```text
e^(-2) = 0.135   (still positive)
```

### 2) Keeps ranking

If logit A > logit B, then `e^A > e^B`. The winner stays the winner.

### 3) Sharpens differences

A 1-point logit gap becomes about a 2.7x weight gap:

```text
e^3 / e^2 = e^(3-2) = e = 2.718...
```

So a token the model strongly prefers gets **much** more probability than a weak alternative.

### 4) Easier to train

Calculus with `e^x` is clean (`d/dx e^x = e^x`), which helps when the model learns from data.

You do not need calculus to use an LLM. This just explains why softmax is standard.

---

## 8. Picking the next token: greedy vs sampling

After softmax, the model has a probability for every token. How does it choose one?

### Greedy decoding (always pick the top token)

| Token | Probability |
|-------|------------:|
| hotpot | 70.5% |
| noodles | 26.0% |
| car | 3.5% |

Greedy always picks **hotpot**.

- **Pros:** stable, repeatable
- **Cons:** can feel repetitive or "safe"

### Random sampling (roll weighted dice)

Same probabilities, different behavior:

- often picks hotpot (~70% of the time)
- sometimes picks noodles (~26%)
- rarely picks car (~3.5%)

Imagine a bag of tickets: 705 hotpot tickets, 260 noodles tickets, 35 car tickets. Draw one ticket at random. That is sampling.

- **Pros:** more varied, creative text
- **Cons:** less predictable; same prompt can give different answers

Real systems (including llama.cpp) also use knobs like **temperature**, **top-k**, and **top-p** to control randomness. Those adjust logits or probabilities before sampling. See [Internal Principles - Sampling](01-llama-cpp-internal-principles.md#step-e-sampling) and [Next-Token Inference Math](02-next-token-inference-math.md) for details.

---

## 9. Common beginner questions

### "If hotpot is 70%, why didn't I get hotpot?"

Because sampling is random unless you use greedy decoding. 70% is likely, not guaranteed.

### "Are logits the same as probabilities?"

No. Logits are raw scores. Softmax converts them into probabilities.

### "Does the model see words like 'food' and 'vehicle' explicitly?"

Not as human labels. It learns numeric patterns from huge text datasets. Our 3-number toy vectors are a teaching simplification.

### "How many tokens does it score each step?"

All of them in the vocabulary - often tens of thousands. That is why the last step (softmax + sampling) matters for speed and memory.

---

## 10. End-to-end recap

At each generation step:

1. Read the context so far.
2. Build a context representation (hidden state).
3. Score every possible next token -> **logits**.
4. Convert logits with **softmax** -> **probabilities** (sum to 100%).
5. Choose one token (greedy or sample).
6. Append it and repeat.

One-sentence version:

> A **logit** is the model's raw preference score for a token; **softmax** turns all logits into probabilities; then the sampler picks the next token.

---

## 11. What to read next

| If you want... | Read... |
|----------------|---------|
| GGUF, KV cache, llama.cpp code layout | [llama.cpp Internal Principles](01-llama-cpp-internal-principles.md) |
| Attention, RoPE, full softmax math in code | [Next-Token Inference Math](02-next-token-inference-math.md) |

You now have the core mental model for the heart of text generation: **logits -> softmax -> sample**.

# Next-Token Inference Math in llama.cpp

The math llama.cpp uses to predict the next token from history. This post uses the **LLaMA-family decoder** as the reference (`src/models/llama.cpp`). Other models (Mamba, RWKV, MoE, etc.) change parts of this, but the overall autoregressive idea is the same.

---

## 0. The core probabilistic model

An LLM is an **autoregressive** model:

{% math %}
P(x_1, x_2, \ldots, x_T) = \prod_{{ '{' }}t=1{{ '}' }}^{{ '{' }}T{{ '}' }} P(x_t \mid x_1, \ldots, x_{{ '{' }}t-1{{ '}' }})
{% endmath %}

At inference time, given history tokens {% math %}x_{{ '{' }}1:t-1{{ '}' }}{% endmath %}, llama.cpp computes:

{% math %}
P(x_t \mid x_{{ '{' }}1:t-1{{ '}' }}) = \text{{ '{' }}softmax{{ '}' }}(z_t)
{% endmath %}

where {% math %}z_t \in \mathbb{{ '{' }}R{{ '}' }}^{{ '{' }}|V|{{ '}' }}{% endmath %} is the **logit vector** over vocabulary {% math %}V{% endmath %}, then **samples** one token from that distribution.

### What is softmax({% math %}z_t{% endmath %})?

Think of {% math %}z_t{% endmath %} as a long list of **raw scores**, one per possible next token. After the transformer runs, llama.cpp has a vector like:

{% math %}
z_t = (z_{{ '{' }}t,1{{ '}' }}, z_{{ '{' }}t,2{{ '}' }}, \ldots, z_{{ '{' }}t,|V|{{ '}' }})
{% endmath %}

Each entry {% math %}z_{{ '{' }}t,v{{ '}' }}{% endmath %} is a **logit**: an unbounded real number saying how plausible token {% math %}v{% endmath %} is as the next token. Higher logit = more likely, but logits are **not** probabilities yet. They can be negative, greater than 1, and they do **not** sum to 1.

**Softmax** converts logits into a valid probability distribution:

{% math %}
p_{{ '{' }}t,v{{ '}' }} = \text{{ '{' }}softmax{{ '}' }}(z_t)_v = \frac{{ '{' }}\exp(z_{{ '{' }}t,v{{ '}' }}){{ '}' }}{{ '{' }}\sum_{{ '{' }}u \in V{{ '}' }} \exp(z_{{ '{' }}t,u{{ '}' }}){{ '}' }}
{% endmath %}

So for every vocabulary token {% math %}v{% endmath %}:

- {% math %}p_{{ '{' }}t,v{{ '}' }} \ge 0{% endmath %} (non-negative)
- {% math %}\sum_{{ '{' }}v \in V{{ '}' }} p_{{ '{' }}t,v{{ '}' }} = 1{% endmath %} (sums to 100%)
- larger {% math %}z_{{ '{' }}t,v{{ '}' }}{% endmath %} implies larger {% math %}p_{{ '{' }}t,v{{ '}' }}{% endmath %}

That is why we can write:

{% math %}
P(x_t = v \mid x_{{ '{' }}1:t-1{{ '}' }}) = p_{{ '{' }}t,v{{ '}' }} = \text{{ '{' }}softmax{{ '}' }}(z_t)_v
{% endmath %}

In words: **the probability of picking token {% math %}v{% endmath %} next equals the softmax probability of logit {% math %}z_{{ '{' }}t,v{{ '}' }}{% endmath %}.**

#### Tiny numeric example

Suppose the vocabulary has only 3 tokens and the model outputs:

{% math %}
z_t = (2.0,\; 1.0,\; 0.1)
{% endmath %}

Compute exponentials: {% math %}e^{{ '{' }}2.0{{ '}' }} \approx 7.39{% endmath %}, {% math %}e^{{ '{' }}1.0{{ '}' }} \approx 2.72{% endmath %}, {% math %}e^{{ '{' }}0.1{{ '}' }} \approx 1.11{% endmath %}. Sum {% math %}\approx 11.22{% endmath %}.

{% math %}
\text{{ '{' }}softmax{{ '}' }}(z_t) \approx (0.66,\; 0.24,\; 0.10)
{% endmath %}

Token 1 has the highest logit (2.0), so it gets about 66% probability. Token 2 gets 24%, token 3 gets 10%. The model would **usually** pick token 1, but not always - that depends on sampling (Section 5).

#### Intuition: why exponentiate?

Softmax uses {% math %}\exp(\cdot){% endmath %} for two practical reasons:

1. **Turn scores into positive weights** - {% math %}\exp(z) > 0{% endmath %} always, so every token keeps some chance.
2. **Amplify differences** - if one logit is much larger than others, its probability dominates. If logits are close, probabilities are more balanced.

So softmax is the bridge from "model raw preference scores" to "legal probabilities we can sample from."

#### Where this happens in llama.cpp

1. **Transformer output** produces logits {% math %}z_t{% endmath %} via the output head (`res->t_logits` in `src/models/llama.cpp`).
2. **Sampler** applies temperature, optional top-k/top-p, then softmax (`llama_sampler_softmax_impl` in `src/llama-sampler.cpp`).
3. **Sample step** draws one token from the resulting distribution.

See Section 5 for temperature, stable softmax, and sampling details.


{% math %}
\text{{ '{' }}history tokens{{ '}' }} \;\rightarrow\; \text{{ '{' }}hidden state {{ '}' }} h_{{ '{' }}t-1{{ '}' }} \;\rightarrow\; \text{{ '{' }}logits {{ '}' }} z_t \;\rightarrow\; \text{{ '{' }}probabilities {{ '}' }} p_t \;\rightarrow\; \text{{ '{' }}sample {{ '}' }} x_t
{% endmath %}

---

## 1. Token embedding (history -> vectors)

Given token IDs {% math %}x_1, \ldots, x_{{ '{' }}t-1{{ '}' }}{% endmath %}, llama.cpp looks up rows from the embedding matrix {% math %}E \in \mathbb{{ '{' }}R{{ '}' }}^{{ '{' }}d \times |V|{{ '}' }}{% endmath %}:

{% math %}
h_i^{{ '{' }}(0){{ '}' }} = E_{{ '{' }}:, x_i{{ '}' }} \in \mathbb{{ '{' }}R{{ '}' }}^d
{% endmath %}

In code this is `ggml_get_rows(tok_embd, ...)` in `src/models/llama.cpp`.

{% math %}d{% endmath %} = `n_embd` (hidden size, e.g. 4096).

---

## 2. One transformer layer (repeated {% math %}L{% endmath %} times)

LLaMA uses **pre-norm** blocks. For layer {% math %}\ell{% endmath %}:

### 2.1 RMSNorm (before attention)

{% math %}
\bar{{ '{' }}h{{ '}' }}_i = \text{{ '{' }}RMSNorm{{ '}' }}(h_i^{{ '{' }}(\ell){{ '}' }}) \odot \gamma_{{ '{' }}\text{{ '{' }}attn{{ '}' }}{{ '}' }}
{% endmath %}

{% math %}
\text{{ '{' }}RMSNorm{{ '}' }}(x) = \frac{{ '{' }}x{{ '}' }}{{ '{' }}\sqrt{{ '{' }}\frac{{ '{' }}1{{ '}' }}{{ '{' }}d{{ '}' }}\sum_{{ '{' }}j=1{{ '}' }}^{{ '{' }}d{{ '}' }} x_j^2 + \epsilon{{ '}' }}{{ '}' }}
{% endmath %}

This matches `ggml_rms_norm` in `ggml/src/ggml-cpu/ops.cpp` (`build_norm` with `LLM_NORM_RMS` in `src/llama-graph.cpp`).

### 2.2 Q, K, V projections

For each token position {% math %}i{% endmath %}:

{% math %}
Q_i = W_Q \bar{{ '{' }}h{{ '}' }}_i,\quad
K_i = W_K \bar{{ '{' }}h{{ '}' }}_i,\quad
V_i = W_V \bar{{ '{' }}h{{ '}' }}_i
{% endmath %}

With multi-head attention, each head {% math %}h{% endmath %} has its own subspace of dimension {% math %}d_h{% endmath %}.

In code: `build_qkv()` then `ggml_rope_ext()` on Q and K.

### 2.3 RoPE (positional encoding)

RoPE rotates pairs of dimensions in Q and K based on position {% math %}p_i{% endmath %}.

For each pair {% math %}(2k, 2k+1){% endmath %}:

{% math %}
\begin{{ '{' }}pmatrix{{ '}' }} q'_{{ '{' }}2k{{ '}' }} \\ q'_{{ '{' }}2k+1{{ '}' }} \end{{ '{' }}pmatrix{{ '}' }}
=
\begin{{ '{' }}pmatrix{{ '}' }}
\cos(p_i \theta_k) & -\sin(p_i \theta_k) \\
\sin(p_i \theta_k) & \cos(p_i \theta_k)
\end{{ '{' }}pmatrix{{ '}' }}
\begin{{ '{' }}pmatrix{{ '}' }} q_{{ '{' }}2k{{ '}' }} \\ q_{{ '{' }}2k+1{{ '}' }} \end{{ '{' }}pmatrix{{ '}' }}
{% endmath %}

Same for {% math %}K{% endmath %}. {% math %}\theta_k{% endmath %} depends on base frequency (`freq_base`, `freq_scale`).

This is `ggml_rope_ext` in `src/models/llama.cpp`.

### 2.4 Causal self-attention

For the **current** query position {% math %}t{% endmath %} (the token we want to predict next), attention over all past positions {% math %}j \le t{% endmath %}:

**Attention scores:**

{% math %}
s_{{ '{' }}t,j{{ '}' }} = \frac{{ '{' }}Q_t^\top K_j{{ '}' }}{{ '{' }}\sqrt{{ '{' }}d_h{{ '}' }}{{ '}' }} \cdot \text{{ '{' }}scale{{ '}' }}
{% endmath %}

Default scale from `src/models/llama.cpp`:

{% math %}
\text{{ '{' }}scale{{ '}' }} = \frac{{ '{' }}1{{ '}' }}{{ '{' }}\sqrt{{ '{' }}d_h{{ '}' }}{{ '}' }}
{% endmath %}

**Causal mask** (cannot attend to future tokens):

{% math %}
\tilde{{ '{' }}s{{ '}' }}_{{ '{' }}t,j{{ '}' }} =
\begin{{ '{' }}cases{{ '}' }}
s_{{ '{' }}t,j{{ '}' }} & j \le t \\
-\infty & j > t
\end{{ '{' }}cases{{ '}' }}
{% endmath %}

**Attention weights:**

{% math %}
\alpha_{{ '{' }}t,j{{ '}' }} = \text{{ '{' }}softmax{{ '}' }}_j(\tilde{{ '{' }}s{{ '}' }}_{{ '{' }}t,j{{ '}' }})
{% endmath %}

In code (`src/llama-graph.cpp`):

```cpp
ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
kq = ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale, ...);
ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);
```

So mathematically:

{% math %}
\text{{ '{' }}Attn{{ '}' }}_t = \sum_{{ '{' }}j \le t{{ '}' }} \alpha_{{ '{' }}t,j{{ '}' }} V_j
{% endmath %}

**Output projection:**

{% math %}
a_t = W_O \cdot \text{{ '{' }}Attn{{ '}' }}_t
{% endmath %}

**Residual:**

{% math %}
u_t = h_t^{{ '{' }}(\ell){{ '}' }} + a_t
{% endmath %}

### 2.5 Feed-forward (SwiGLU in LLaMA)

Another RMSNorm, then SwiGLU FFN:

{% math %}
\bar{{ '{' }}u{{ '}' }}_t = \text{{ '{' }}RMSNorm{{ '}' }}(u_t) \odot \gamma_{{ '{' }}\text{{ '{' }}ffn{{ '}' }}{{ '}' }}
{% endmath %}

{% math %}
\text{{ '{' }}gate{{ '}' }}_t = W_{{ '{' }}\text{{ '{' }}gate{{ '}' }}{{ '}' }} \bar{{ '{' }}u{{ '}' }}_t,\quad
\text{{ '{' }}up{{ '}' }}_t = W_{{ '{' }}\text{{ '{' }}up{{ '}' }}{{ '}' }} \bar{{ '{' }}u{{ '}' }}_t
{% endmath %}

{% math %}
\text{{ '{' }}SiLU{{ '}' }}(x) = x \cdot \sigma(x) = \frac{{ '{' }}x{{ '}' }}{{ '{' }}1 + e^{{ '{' }}-x{{ '}' }}{{ '}' }}
{% endmath %}

{% math %}
\text{{ '{' }}ffn{{ '}' }}_t = W_{{ '{' }}\text{{ '{' }}down{{ '}' }}{{ '}' }} \big(\text{{ '{' }}SiLU{{ '}' }}(\text{{ '{' }}gate{{ '}' }}_t) \odot \text{{ '{' }}up{{ '}' }}_t\big)
{% endmath %}

This is exactly what `ggml_vec_swiglu_f32` computes:

{% math %}
y = \text{{ '{' }}SiLU{{ '}' }}(x) \odot g
{% endmath %}

from `ggml/src/ggml-cpu/vec.cpp`.

**Second residual:**

{% math %}
h_t^{{ '{' }}(\ell+1){{ '}' }} = u_t + \text{{ '{' }}ffn{{ '}' }}_t
{% endmath %}

Repeat for all {% math %}L{% endmath %} layers.

---

## 3. KV cache: how history is reused efficiently

Naively, recomputing attention for token {% math %}t{% endmath %} would recompute all {% math %}K_j, V_j{% endmath %} for {% math %}j < t{% endmath %}. That is too slow.

llama.cpp stores past keys/values in the **KV cache** (`src/llama-kv-cache.h`). At decode time for the new token:

- compute only {% math %}Q_t, K_t, V_t{% endmath %} for the new position
- read cached {% math %}K_{{ '{' }}1:t-1{{ '}' }}, V_{{ '{' }}1:t-1{{ '}' }}{% endmath %}
- attention uses all {% math %}j \le t{% endmath %}

So the math is unchanged, but compute drops from {% math %}O(t^2){% endmath %} per step to roughly {% math %}O(t){% endmath %} per step (with flash attention, memory access is also optimized).

---

## 4. Output head: hidden state -> logits

After the final layer:

{% math %}
\tilde{{ '{' }}h{{ '}' }}_t = \text{{ '{' }}RMSNorm{{ '}' }}(h_t^{{ '{' }}(L){{ '}' }}) \odot \gamma_{{ '{' }}\text{{ '{' }}out{{ '}' }}{{ '}' }}
{% endmath %}

{% math %}
z_t = W_{{ '{' }}\text{{ '{' }}out{{ '}' }}{{ '}' }} \tilde{{ '{' }}h{{ '}' }}_t \in \mathbb{{ '{' }}R{{ '}' }}^{{ '{' }}|V|{{ '}' }}
{% endmath %}

From `src/models/llama.cpp`:

```cpp
cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
cur = build_lora_mm(model.output, cur, model.output_s);
res->t_logits = cur;
```

{% math %}z_t[v]{% endmath %} is the **logit** for vocabulary token {% math %}v{% endmath %}.

Often {% math %}W_{{ '{' }}\text{{ '{' }}out{{ '}' }}{{ '}' }} = E^\top{% endmath %} (weight tying), so the same embedding matrix is reused.

---

## 5. Turning logits into the next token (sampling)

Let logits be {% math %}z = (z_1, \ldots, z_{{ '{' }}|V|{{ '}' }}){% endmath %}.

### 5.1 Temperature

{% math %}
\tilde{{ '{' }}z{{ '}' }}_v = \frac{{ '{' }}z_v{{ '}' }}{{ '{' }}T{{ '}' }}
{% endmath %}

If {% math %}T \to 0{% endmath %}, this becomes greedy argmax (`llama_sampler_temp_impl` in `src/llama-sampler.cpp`).

### 5.2 Softmax (stable form used in code)

{% math %}
p_v = \frac{{ '{' }}\exp(\tilde{{ '{' }}z{{ '}' }}_v - \max_u \tilde{{ '{' }}z{{ '}' }}_u){{ '}' }}{{ '{' }}\sum_{{ '{' }}u{{ '}' }} \exp(\tilde{{ '{' }}z{{ '}' }}_u - \max_u \tilde{{ '{' }}z{{ '}' }}_u){{ '}' }}
{% endmath %}

From `llama_sampler_softmax_impl` in `src/llama-sampler.cpp`:

```cpp
float p = expf(cur_p->data[i].logit - max_l);
cur_p->data[i].p = p / cum_sum;
```

### 5.3 Optional filters

**Top-k:** keep only the {% math %}k{% endmath %} largest logits.

**Top-p (nucleus):** keep smallest set {% math %}S{% endmath %} such that:

{% math %}
\sum_{{ '{' }}v \in S{{ '}' }} p_v \ge p_{{ '{' }}\text{{ '{' }}threshold{{ '}' }}{{ '}' }}
{% endmath %}

### 5.4 Sample

Draw next token:

{% math %}
x_t \sim \text{{ '{' }}Categorical{{ '}' }}(p)
{% endmath %}

Or greedy:

{% math %}
x_t = \arg\max_v p_v
{% endmath %}

Then append {% math %}x_t{% endmath %} to history and repeat.

---

## 6. Full next-token formula (compact)

Putting it all together, for history {% math %}x_{{ '{' }}1:t-1{{ '}' }}{% endmath %}:

{% math %}
x_t \sim \text{{ '{' }}Sample{{ '}' }}\!\left(
\text{{ '{' }}softmax{{ '}' }}\!\left(
\frac{{ '{' }}
W_{{ '{' }}\text{{ '{' }}out{{ '}' }}{{ '}' }} \cdot
f_{{ '{' }}\theta{{ '}' }}\!\left(x_{{ '{' }}1:t-1{{ '}' }}\right)
{{ '}' }}{{ '{' }}T{{ '}' }}
\right)
\right)
{% endmath %}

where {% math %}f_{{ '{' }}\theta{{ '}' }}{% endmath %} is the full transformer stack:

{% math %}
f_{{ '{' }}\theta{{ '}' }}(x_{{ '{' }}1:t-1{{ '}' }}) =
\text{{ '{' }}RMSNorm{{ '}' }}\!\left(
\text{{ '{' }}Transformer{{ '}' }}^{{ '{' }}(L){{ '}' }}\!\left(
\ldots
\text{{ '{' }}Transformer{{ '}' }}^{{ '{' }}(1){{ '}' }}\!\left(
E[x_{{ '{' }}1:t-1{{ '}' }}]
\right)\ldots
\right)
\right)
{% endmath %}

and each transformer layer is:

{% math %}
x \leftarrow x + \text{{ '{' }}FFN{{ '}' }}(\text{{ '{' }}RMSNorm{{ '}' }}(x))
{% endmath %}
{% math %}
x \leftarrow x + \text{{ '{' }}Attn{{ '}' }}(\text{{ '{' }}RoPE{{ '}' }}(\text{{ '{' }}RMSNorm{{ '}' }}(x)))
{% endmath %}

(attention uses all past positions via KV cache).

---

## 7. Prefill vs decode (same math, different batch shape)

| Phase | Input | What is computed |
|-------|--------|------------------|
| **Prefill** | whole prompt {% math %}x_{{ '{' }}1:t-1{{ '}' }}{% endmath %} at once | all hidden states + fill KV cache |
| **Decode** | only newest token {% math %}x_{{ '{' }}t-1{{ '}' }}{% endmath %} | one new row of logits {% math %}z_t{% endmath %} |

Math is identical; only efficiency differs.

---

## 8. Important caveats

1. **Model-specific variants**: Gemma, Qwen, Mamba, MoE, etc. change norm type, activation, attention style, but the pattern remains: `history -> hidden -> logits -> sample`.
2. **Quantization** (Q4_K, etc.) approximates matrix multiplies with lower-precision weights; formulas are the same, numbers are approximate.
3. **Chat templates** are outside the model math: they format text before tokenization.

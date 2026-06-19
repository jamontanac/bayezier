
### Phase 2 — Metropolis-within-Gibbs (sampling engine)

**Purpose:** Draw posterior samples of (k, β) using MCMC.
**Input:**  `X_train`, `y_train`, `k_candidate_values`, MCMC `config` (n_iters, burn_in, thinning, proposal_width, beta_sigma), plus the precomputed `Count_Tensor` from Phase 1.
**Output:** `chain` — list of (β, k) pairs after burn-in and thinning.
**Structure:** One iteration = Gibbs over k (exhaustive softmax) → MH over β (Gaussian proposal + accept/reject).

**Related docs:**
- [[Precomputation|Phase 1 (precomputation)]] — builds the `Count_Tensor` this phase reads
- [[Inference|Phase 3 (prediction)]] — consumes the chain this phase produces
- [[Steps_for_sampling_parameters|Sampler walkthrough]] — detailed explanation of Gibbs, MH, and the update cycle
- [[Technical_details_sampling_parameters|Sampling parameters]] — how to choose `n_iters`, `burn_in`, `thinning`, `proposal_width`
- [[JointMh_sampler|JointMh sampler]] — alternate sampler that proposes k and β jointly

---

## Hybrid sampler (default)

```
Function Fit_Bayes_KNN(X_train, y_train, k_candidate_values, config):
    beta    = 1.0
    k_index = median_index(k_candidate_values)

    Count_Tensor = Prepare_Static_Structures(...)
    chain = []

    # n_iters = burn_in + n_samples * thinning
    For step in 0..n_iters:

        # --- Gibbs step: sample k from its exact full conditional ---
        For ki in 0..len(k_candidate_values):
            log_w[ki] = Evaluate_Log_Joint(Count_Tensor, y_train, beta, ki)
        k_index = sample_categorical(log_w)        # log-sum-exp normalization inside

        # --- MH step: sample beta using the newly updated k_index ---
        beta_prop = beta + Normal(0, proposal_width).sample()
        If beta_prop > 0:
            log_alpha = Evaluate_Log_Joint(..., beta_prop, k_index)
                      - Evaluate_Log_Joint(..., beta,      k_index)
            If log(Uniform(0,1)) < log_alpha:
                beta = beta_prop
        # beta_prop <= 0: half-normal prior gives -inf; always rejected.

        # --- Record post-burn-in, every `thinning` steps ---
        If step >= burn_in and (step - burn_in) % thinning == 0:
            chain.append((beta, k_values[k_index]))

    Return chain


Function Evaluate_Log_Joint(Count_Tensor, y_train, beta, k_index):
    Return log_likelihood(Count_Tensor, y_train, beta, k_index)
         + log_prior_k(k_index, n_k_candidates)
         + log_prior_beta(beta, beta_sigma)


Function log_likelihood(Count_Tensor, y_train, beta, k_index):
    total = 0.0
    For each point i:
        counts  = Count_Tensor[i, k_index, :]
        k       = k_candidate_values[k_index]
        logits  = beta * counts / k
        log_p_i = logits[y_train[i]] - log_sum_exp(logits)
        total  += log_p_i
    Return total


Function log_prior_k(k_index, n_k_candidates):
    Return -ln(n_k_candidates)          # discrete uniform; constant, cancels in softmax


Function log_prior_beta(beta, sigma):
    If beta <= 0: Return -inf
    Return -ln(sigma) - 0.5 * (beta / sigma)^2 - ln(sqrt(2/pi))   # half-normal


Function sample_categorical(log_weights):
    # Numerically stable: subtract max before exp to avoid underflow.
    lse    = log_sum_exp(log_weights)
    probs  = [exp(lw - lse) for lw in log_weights]
    u      = Uniform(0, 1).sample()
    cumsum = 0.0
    For i, p in enumerate(probs):
        cumsum += p
        If u <= cumsum: Return i
    Return len(probs) - 1    # floating-point rounding fallback
```

---

## JointMh sampler (alternate)

See [[JointMh_sampler]] for the full design rationale.

```
Function Fit_Bayes_KNN_JointMH(X_train, y_train, k_candidate_values, config):
    beta    = 1.0
    k_index = median_index(k_candidate_values)

    Count_Tensor = Prepare_Static_Structures(...)
    chain = []

    For step in 0..n_iters:

        # --- Joint proposal: k and beta simultaneously ---
        k_prop    = Uniform{0, …, n_candidates − 1}.sample()   # independent of current k
        beta_prop = beta + Normal(0, proposal_width).sample()

        # Both proposal ratios cancel (see JointMh_sampler for proof).
        If beta_prop > 0:
            log_alpha = Evaluate_Log_Joint(..., beta_prop, k_prop)
                      - Evaluate_Log_Joint(..., beta,      k_index)
            If log(Uniform(0,1)) < log_alpha:
                k_index = k_prop
                beta    = beta_prop
        # If beta_prop <= 0, reject without evaluation.

        If step >= burn_in and (step - burn_in) % thinning == 0:
            chain.append((beta, k_values[k_index]))

    Return chain
```

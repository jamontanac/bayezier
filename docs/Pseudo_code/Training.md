
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

```
Function Fit_Bayes_KNN(X_train, y_train, k_candidate_values, config):
    beta    = 1.0
    k_index = median_index(k_candidate_values)

    Count_Tensor = Prepare_Static_Structures(...)
    chain = []

    For step in 1..n_iters:

        # --- Gibbs step: sample k ---
        For ki in 0..len(k_candidate_values):
            log_w[ki] = Evaluate_Log_Joint(Count_Tensor, y_train, beta, ki)
        k_probs = softmax_stable(log_w)        # log-sum-exp normalization
        k_index = sample_categorical(k_probs)

        # --- MH step: sample beta ---
        beta_prop = Normal(beta, proposal_width).sample()
        If beta_prop > 0:
            log_alpha = Evaluate_Log_Joint(..., beta_prop, k_index)
                      - Evaluate_Log_Joint(..., beta,      k_index)
            If log(Uniform(0,1)) < log_alpha:
                beta = beta_prop

        # --- Record ---
        If step > burn_in and step % thinning == 0:
            chain.append((beta, k_candidate_values[k_index]))

    Return chain, X_train, y_train


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
```

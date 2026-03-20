"""
OGI Framework - Theorem 4 Benchmark
Corrected version: both conditions train, projection head added,
timing stabilized with more trials.
"""

import torch
import torch.nn as nn
import time
import numpy as np

torch.manual_seed(42)
np.random.seed(42)


class OGIFusionCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # GRU for temporal stability (Lemma 4.2)
        self.gru = nn.GRUCell(input_dim, hidden_dim)

        # Projection head: maps hidden state back to input space
        # so cosine similarity is measured in a meaningful common space
        self.projection = nn.Linear(hidden_dim, input_dim)

        # Critic network for MINE (Lemma 4.1)
        # Takes [h_t || context] -> scalar
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x, h_prev):
        h_t = self.gru(x, h_prev)
        o_t = self.projection(h_t)   # output in input space
        return h_t, o_t

    def coherence_loss(self, h_t, context):
        """
        MINE variational lower bound on I(h_t; context).
        Equation 11 in the paper.
        Joint: (h_t, context) from same sample.
        Marginal: (h_t, shuffled context) - breaks statistical dependence.
        """
        joint    = torch.cat([h_t, context], dim=1)
        shuffled = context[torch.randperm(context.size(0))]
        marginal = torch.cat([h_t, shuffled], dim=1)

        t_joint    = self.critic(joint)
        t_marginal = self.critic(marginal)

        mi_lower_bound = (torch.mean(t_joint)
                          - torch.log(torch.mean(torch.exp(t_marginal)) + 1e-8))
        return -mi_lower_bound  # minimize negative MI = maximize MI

    def task_loss(self, o_t, context):
        """
        Reconstruction loss: encourages projected output to align with context.
        This is the L_task term — both conditions optimize this.
        """
        return nn.functional.mse_loss(o_t, context)


def run_benchmark(enable_coherence=True, trials=500, dim=128, batch_size=32,
                  coherence_weight=0.5):
    """
    Both conditions (baseline and OGI) train on the same task loss.
    OGI additionally optimizes the coherence loss.
    Timing uses wall-clock over the full trial loop for stability.
    """
    model = OGIFusionCell(input_dim=dim, hidden_dim=dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Fixed ground-truth semantic context
    context = torch.randn(batch_size, dim)

    similarities = []
    losses = []

    # Warm-up pass (not timed)
    h_prev = torch.zeros(batch_size, dim)
    noisy_input = context + torch.randn(batch_size, dim) * 0.5
    h_t, o_t = model(noisy_input, h_prev)

    # Timed trials
    h_prev = torch.zeros(batch_size, dim)
    start = time.perf_counter()

    for i in range(trials):
        optimizer.zero_grad()

        # Fresh noisy sample each trial (simulates streaming input)
        noisy_input = context + torch.randn(batch_size, dim) * 0.5

        h_t, o_t = model(noisy_input, h_prev)

        # Both conditions optimize task loss (reconstruction)
        loss = model.task_loss(o_t, context)

        if enable_coherence:
            loss = loss + coherence_weight * model.coherence_loss(h_t, context)

        loss.backward()
        optimizer.step()

        # Cosine similarity in projected (output) space - meaningful comparison
        with torch.no_grad():
            cos_sim = nn.functional.cosine_similarity(o_t, context).mean().item()
            similarities.append(cos_sim)
            losses.append(loss.item())

        h_prev = h_t.detach()

    elapsed = time.perf_counter() - start

    return {
        "mean_similarity": np.mean(similarities),
        "final_similarity": np.mean(similarities[-50:]),  # last 50 trials
        "mean_loss": np.mean(losses),
        "elapsed_s": elapsed,
        "per_trial_ms": (elapsed / trials) * 1000,
        "similarities": similarities
    }


def run_stability_test(trials=200, dim=128, batch_size=32, noise_levels=None):
    """
    Tests temporal stability: semantic binding across cycles
    under increasing Gaussian noise. Validates Lemma 4.2.
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.25, 0.5, 0.75, 1.0]

    print("\n--- Stability Test (Lemma 4.2) ---")
    print(f"{'Noise σ':>10} | {'Final CosSim':>14} | {'Cycles Stable (>0.7)':>20}")
    print("-" * 50)

    for sigma in noise_levels:
        model = OGIFusionCell(input_dim=dim, hidden_dim=dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        context = torch.randn(batch_size, dim)
        h_prev = torch.zeros(batch_size, dim)
        stable_count = 0

        for _ in range(trials):
            optimizer.zero_grad()
            noisy_input = context + torch.randn(batch_size, dim) * sigma
            h_t, o_t = model(noisy_input, h_prev)
            loss = model.task_loss(o_t, context) + 0.5 * model.coherence_loss(h_t, context)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                sim = nn.functional.cosine_similarity(o_t, context).mean().item()
                if sim > 0.7:
                    stable_count += 1
            h_prev = h_t.detach()

        with torch.no_grad():
            final_sim = nn.functional.cosine_similarity(o_t, context).mean().item()

        print(f"{sigma:>10.2f} | {final_sim:>14.4f} | {stable_count:>20d}/{trials}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("OGI Framework — Theorem 4 Benchmark")
    print("Corrected: both conditions train; projection head used")
    print("=" * 60)

    print("\nRunning baseline (task loss only, no coherence)...")
    base = run_benchmark(enable_coherence=False, trials=500)

    print("Running OGI (task loss + coherence objective)...")
    ogi  = run_benchmark(enable_coherence=True,  trials=500)

    sim_gain     = ((ogi["final_similarity"] - base["final_similarity"])
                    / abs(base["final_similarity"]) * 100)
    time_overhead = ((ogi["elapsed_s"] - base["elapsed_s"])
                     / base["elapsed_s"] * 100)
    ms_delta      = ogi["per_trial_ms"] - base["per_trial_ms"]

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"{'':30} {'Baseline':>12} {'OGI':>12}")
    print(f"{'Mean cosine similarity':30} {base['mean_similarity']:>12.4f} {ogi['mean_similarity']:>12.4f}")
    print(f"{'Final cosine similarity':30} {base['final_similarity']:>12.4f} {ogi['final_similarity']:>12.4f}")
    print(f"{'Per-trial time (ms)':30} {base['per_trial_ms']:>12.2f} {ogi['per_trial_ms']:>12.2f}")
    print(f"{'Total elapsed (s)':30} {base['elapsed_s']:>12.3f} {ogi['elapsed_s']:>12.3f}")
    print()
    print(f"Semantic similarity gain (final 50 trials): {sim_gain:+.1f}%")
    print(f"Compute overhead (Coherence Tax):           {time_overhead:+.1f}%")
    print(f"Per-trial ms delta:                         {ms_delta:+.2f}ms")

    run_stability_test(trials=200, noise_levels=[0.0, 0.25, 0.5, 0.75, 1.0])

    print("\nDone. These numbers are what goes in the paper.")
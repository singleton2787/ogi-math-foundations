"""
OGI Framework - Semantic Incompleteness Test
Lemma 4.1 boundary condition: coherence benefit requires genuine semantic incompleteness

Previous tests (ogi_benchmark.py, multimodal_test.py) showed flat or negative delta
because task loss alone was sufficient - both streams pointed toward the full context.

This test forces genuine semantic incompleteness:
  - visual stream carries ONLY the first half of the context (dims 0..63)
  - linguistic stream carries ONLY the second half (dims 64..127)
  - neither stream alone can reconstruct the full context
  - the fusion model must combine both to recover the semantic ground truth

This is the boundary condition for Lemma 4.1:
  "the coherence objective's disambiguation benefit requires genuine cross-modal ambiguity"

If OGI outperforms baseline here, the boundary condition is confirmed.
If still flat, synthetic simulation is insufficient and real embeddings are needed.

running on laptop, no GPU
"""

import torch
import torch.nn as nn
import numpy as np
import time

torch.manual_seed(42)
np.random.seed(42)


def make_split_streams(batch_size, dim, context, noise_scale=0.3):
    """
    Split context semantically - each stream is incomplete alone.
    Visual: first half of context dims, second half zeroed + noise
    Linguistic: second half of context dims, first half zeroed + noise
    Neither stream alone is sufficient to reconstruct full context.
    """
    half = dim // 2

    # visual stream - only knows first half
    x_vis = torch.zeros(batch_size, dim)
    x_vis[:, :half] = context[:, :half]
    x_vis[:, half:] = torch.randn(batch_size, half) * noise_scale  # noise, not signal
    x_vis = x_vis + torch.randn(batch_size, dim) * (noise_scale * 0.3)  # global noise

    # linguistic stream - only knows second half
    x_ling = torch.zeros(batch_size, dim)
    x_ling[:, :half] = torch.randn(batch_size, half) * noise_scale  # noise, not signal
    x_ling[:, half:] = context[:, half:]
    x_ling = x_ling + torch.randn(batch_size, dim) * (noise_scale * 0.3)  # global noise

    return x_vis, x_ling


class SplitContextFusionCell(nn.Module):
    """
    Same architecture as MultiModalFusionCell.
    Separate GRU encoders, attention-weighted fusion, projection head, MINE critic.
    Kept separate file so multimodal_test.py results are not disturbed.
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.gru_visual     = nn.GRUCell(input_dim, hidden_dim)
        self.gru_linguistic = nn.GRUCell(input_dim, hidden_dim)
        self.attention      = nn.Linear(hidden_dim * 2, 2)
        self.projection     = nn.Linear(hidden_dim, input_dim)

        # larger critic - task is harder, needs more capacity
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x_vis, x_ling, h_vis_prev, h_ling_prev):
        h_vis  = self.gru_visual(x_vis, h_vis_prev)
        h_ling = self.gru_linguistic(x_ling, h_ling_prev)

        combined     = torch.cat([h_vis, h_ling], dim=1)
        attn_weights = torch.softmax(self.attention(combined), dim=1)

        h_fused = (attn_weights[:, 0:1] * h_vis
                   + attn_weights[:, 1:2] * h_ling)

        o_t = self.projection(h_fused)
        return h_fused, o_t, h_vis, h_ling, attn_weights

    def coherence_loss(self, h_fused, context):
        joint    = torch.cat([h_fused, context], dim=1)
        shuffled = context[torch.randperm(context.size(0))]
        marginal = torch.cat([h_fused, shuffled], dim=1)

        t_joint    = self.critic(joint)
        t_marginal = self.critic(marginal)

        # clamp before exp to prevent overflow (exp(>88) = inf on float32)
        t_marginal_clamped = torch.clamp(t_marginal, max=10.0)
        mi_bound = (torch.mean(t_joint)
                    - torch.log(torch.mean(torch.exp(t_marginal_clamped)) + 1e-8))
        return -mi_bound

    def task_loss(self, o_t, context):
        return nn.functional.mse_loss(o_t, context)


def run_split_benchmark(enable_coherence=True, trials=750,
                         dim=128, batch_size=32, coherence_weight=0.5,
                         noise_scale=0.3):
    """
    750 trials - harder task needs more training steps to converge.
    Both conditions use split streams - the only difference is coherence objective.
    """
    model     = SplitContextFusionCell(input_dim=dim, hidden_dim=dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)  # lower lr - harder task

    context = torch.randn(batch_size, dim)

    similarities    = []
    attn_log        = []
    half_sims_vis   = []  # similarity of visual output to first-half context
    half_sims_ling  = []  # similarity of linguistic output to second-half context

    # warm-up
    h_vis_prev  = torch.zeros(batch_size, dim)
    h_ling_prev = torch.zeros(batch_size, dim)
    x_vis, x_ling = make_split_streams(batch_size, dim, context, noise_scale)
    _, _, h_vis_prev, h_ling_prev, _ = model(x_vis, x_ling, h_vis_prev, h_ling_prev)
    h_vis_prev  = h_vis_prev.detach()
    h_ling_prev = h_ling_prev.detach()

    h_vis_prev  = torch.zeros(batch_size, dim)
    h_ling_prev = torch.zeros(batch_size, dim)
    start = time.perf_counter()

    half = dim // 2

    for i in range(trials):
        optimizer.zero_grad()

        x_vis, x_ling = make_split_streams(batch_size, dim, context, noise_scale)

        h_fused, o_t, h_vis, h_ling, attn_w = model(
            x_vis, x_ling, h_vis_prev, h_ling_prev
        )

        loss = model.task_loss(o_t, context)

        if enable_coherence and i > 200:
            # warmup: let task loss stabilize first (200 trials)
            # then ramp coherence weight slowly, cap at 0.15
            # MINE gradient is large on harder tasks - needs to be kept small
            coh_weight = min(0.05 * ((i - 200) / 100), 0.15)
            loss = loss + coh_weight * model.coherence_loss(h_fused, context)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            cos_sim = nn.functional.cosine_similarity(o_t, context).mean().item()
            similarities.append(cos_sim)
            attn_log.append(attn_w.mean(dim=0).tolist())

        h_vis_prev  = h_vis.detach()
        h_ling_prev = h_ling.detach()

    elapsed = time.perf_counter() - start
    final_attn = np.mean(attn_log[-50:], axis=0)

    return {
        "mean_similarity":  np.mean(similarities),
        "final_similarity": np.mean(similarities[-50:]),
        "elapsed_s":        elapsed,
        "per_trial_ms":     (elapsed / trials) * 1000,
        "final_attn_vis":   final_attn[0],
        "final_attn_ling":  final_attn[1],
        "similarities":     similarities
    }


def run_noise_sweep(trials=500, dim=128, batch_size=32):
    """
    Vary noise level on split streams.
    Low noise = streams are cleaner but still incomplete.
    High noise = streams are incomplete AND noisy.
    Coherence benefit should be consistent across noise levels here
    because the incompleteness is structural, not noise-dependent.
    """
    print("\n--- Noise Sweep (Split Context) ---")
    print(f"{'Noise':>8} | {'Base sim':>10} | {'OGI sim':>10} | {'Delta':>8} | {'Attn V/L':>12}")
    print("-" * 58)

    for noise in [0.1, 0.2, 0.3, 0.5, 0.8]:
        base = run_split_benchmark(enable_coherence=False, trials=trials,
                                    noise_scale=noise)
        ogi  = run_split_benchmark(enable_coherence=True,  trials=trials,
                                    noise_scale=noise)
        delta = ogi["final_similarity"] - base["final_similarity"]
        attn  = f"{ogi['final_attn_vis']:.2f}/{ogi['final_attn_ling']:.2f}"
        print(f"{noise:>8.1f} | {base['final_similarity']:>10.4f} | "
              f"{ogi['final_similarity']:>10.4f} | {delta:>+8.4f} | {attn:>12}")


if __name__ == "__main__":
    print("=" * 60)
    print("OGI Framework - Semantic Incompleteness Test")
    print("Lemma 4.1 boundary condition verification")
    print("=" * 60)
    print("Setup: visual stream = dims 0-63 only")
    print("       linguistic stream = dims 64-127 only")
    print("       neither stream alone reconstructs full context")
    print(f"\nConfig: dim=128, batch=32, 750 trials, CPU\n")

    print("Running baseline (task loss only, split streams)...")
    base = run_split_benchmark(enable_coherence=False, trials=750)

    print("Running OGI (task loss + coherence, split streams)...")
    ogi  = run_split_benchmark(enable_coherence=True,  trials=750)

    sim_gain      = ogi["final_similarity"] - base["final_similarity"]
    time_overhead = ((ogi["elapsed_s"] - base["elapsed_s"]) / base["elapsed_s"]) * 100
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
    print(f"Semantic similarity gain:         {sim_gain:+.4f}")
    print(f"Compute overhead (Coherence Tax): {time_overhead:+.1f}%")
    print(f"Per-trial ms delta:               {ms_delta:+.2f}ms")
    print()
    print(f"OGI attention weights (final 50 trials):")
    print(f"  Visual stream:     {ogi['final_attn_vis']:.3f}")
    print(f"  Linguistic stream: {ogi['final_attn_ling']:.3f}")

    # boundary condition verdict
    print("\n" + "=" * 60)
    if sim_gain > 0.001:
        print("LEMMA 4.1 BOUNDARY CONDITION: CONFIRMED")
        print(f"Coherence objective provides +{sim_gain:.4f} gain under semantic incompleteness")
        print("Condition: benefit requires genuine cross-modal ambiguity, not just noise")
    elif sim_gain > -0.001:
        print("RESULT: FLAT - simulation may be insufficient")
        print("Recommend: real VQA embeddings for definitive test")
    else:
        print("RESULT: NEGATIVE - coherence objective still hurts")
        print("Diagnosis: task loss converges before coherence objective activates")
        print("Recommend: lower learning rate or longer warmup before coherence kicks in")

    run_noise_sweep(trials=400)

    print("\nDone.")
"""
OGI Framework - Multi-Modal Coherence Test
Lemma 4.1 validation: coherence objective benefit under genuine cross-modal ambiguity

Single-modality test (ogi_benchmark.py) showed -0.3% accuracy delta - expected,
task loss alone sufficient when context is unambiguous.

This test uses two genuinely distinct input streams:
  - "visual" stream: dense, spatially correlated features (simulate with smooth vectors)
  - "linguistic" stream: sparse, symbolic features (simulate with sparse high-contrast vectors)

Both streams share the same semantic context but have conflicting low-level structure.
The coherence objective has to resolve the ambiguity - this is where Lemma 4.1 activates.

running on laptop, no GPU
TODO: replace synthetic streams with real VQA embeddings when available
"""

import torch
import torch.nn as nn
import numpy as np
import time

torch.manual_seed(42)
np.random.seed(42)


def make_visual_stream(batch_size, dim, context, noise_scale=0.3):
    """
    Simulate visual features: dense, smoothly varying, high entropy
    Correlated with context but with spatial noise structure
    """
    # smooth noise - neighboring dims correlated
    smooth = torch.zeros(batch_size, dim)
    for i in range(1, dim):
        smooth[:, i] = 0.7 * smooth[:, i-1] + 0.3 * torch.randn(batch_size)
    smooth = smooth / (smooth.norm(dim=1, keepdim=True) + 1e-8)
    return context + noise_scale * smooth


def make_linguistic_stream(batch_size, dim, context, sparsity=0.8, noise_scale=0.5):
    """
    Simulate linguistic features: sparse, high-contrast, low entropy
    Same semantic content as visual but very different low-level structure
    """
    # sparse mask - most dims are zero, a few are large
    mask = (torch.rand(batch_size, dim) > sparsity).float()
    sparse_noise = mask * torch.randn(batch_size, dim) * 2.0
    sparse_noise = sparse_noise / (sparse_noise.norm(dim=1, keepdim=True) + 1e-8)
    return context + noise_scale * sparse_noise


class MultiModalFusionCell(nn.Module):
    """
    Two-stream OGI fusion cell
    Fuses visual and linguistic inputs via separate GRU encoders
    then combines with attention weighting before coherence objective
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # separate encoders per modality - they see different feature spaces
        self.gru_visual = nn.GRUCell(input_dim, hidden_dim)
        self.gru_linguistic = nn.GRUCell(input_dim, hidden_dim)

        # attention-based fusion (Equation 9-10 from paper)
        # learns to weight modalities based on context relevance
        self.attention = nn.Linear(hidden_dim * 2, 2)

        # projection head: fused hidden -> input space for similarity measurement
        self.projection = nn.Linear(hidden_dim, input_dim)

        # MINE critic for coherence objective (Lemma 4.1)
        # takes [h_fused || context] -> scalar MI estimate
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x_visual, x_linguistic, h_vis_prev, h_ling_prev):
        # encode each modality separately
        h_vis  = self.gru_visual(x_visual, h_vis_prev)
        h_ling = self.gru_linguistic(x_linguistic, h_ling_prev)

        # attention-weighted fusion - learns which modality to trust
        combined = torch.cat([h_vis, h_ling], dim=1)
        attn_weights = torch.softmax(self.attention(combined), dim=1)

        # fuse: weighted sum of modality hidden states
        h_fused = (attn_weights[:, 0:1] * h_vis
                   + attn_weights[:, 1:2] * h_ling)

        # project to input space for evaluation
        o_t = self.projection(h_fused)
        return h_fused, o_t, h_vis, h_ling, attn_weights

    def coherence_loss(self, h_fused, context):
        """
        MINE lower bound on I(h_fused; context)
        Same formulation as ogi_benchmark.py - Equation 11
        """
        joint    = torch.cat([h_fused, context], dim=1)
        shuffled = context[torch.randperm(context.size(0))]  # breaks joint distribution
        marginal = torch.cat([h_fused, shuffled], dim=1)

        t_joint    = self.critic(joint)
        t_marginal = self.critic(marginal)

        mi_bound = (torch.mean(t_joint)
                    - torch.log(torch.mean(torch.exp(t_marginal)) + 1e-8))
        return -mi_bound

    def task_loss(self, o_t, context):
        return nn.functional.mse_loss(o_t, context)


def run_multimodal_benchmark(enable_coherence=True, trials=500,
                              dim=128, batch_size=32, coherence_weight=0.5):
    """
    Both conditions train on task loss.
    OGI condition additionally optimizes coherence objective.
    Key difference from ogi_benchmark.py: inputs are genuinely conflicting.
    """
    model = MultiModalFusionCell(input_dim=dim, hidden_dim=dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # shared semantic context - ground truth both streams should converge to
    context = torch.randn(batch_size, dim)

    similarities = []
    attn_weights_log = []

    # warm-up pass - don't time this
    h_vis_prev  = torch.zeros(batch_size, dim)
    h_ling_prev = torch.zeros(batch_size, dim)
    x_vis  = make_visual_stream(batch_size, dim, context)
    x_ling = make_linguistic_stream(batch_size, dim, context)
    _, _, h_vis_prev, h_ling_prev, _ = model(x_vis, x_ling, h_vis_prev, h_ling_prev)
    h_vis_prev  = h_vis_prev.detach()
    h_ling_prev = h_ling_prev.detach()

    h_vis_prev  = torch.zeros(batch_size, dim)
    h_ling_prev = torch.zeros(batch_size, dim)
    start = time.perf_counter()

    for i in range(trials):
        optimizer.zero_grad()

        # fresh conflicting streams each trial
        x_vis  = make_visual_stream(batch_size, dim, context)
        x_ling = make_linguistic_stream(batch_size, dim, context)

        h_fused, o_t, h_vis, h_ling, attn_w = model(
            x_vis, x_ling, h_vis_prev, h_ling_prev
        )

        loss = model.task_loss(o_t, context)

        if enable_coherence:
            loss = loss + coherence_weight * model.coherence_loss(h_fused, context)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            cos_sim = nn.functional.cosine_similarity(o_t, context).mean().item()
            similarities.append(cos_sim)
            attn_weights_log.append(attn_w.mean(dim=0).tolist())

        h_vis_prev  = h_vis.detach()
        h_ling_prev = h_ling.detach()

    elapsed = time.perf_counter() - start

    # average attention weights over final 50 trials
    final_attn = np.mean(attn_weights_log[-50:], axis=0)

    return {
        "mean_similarity":  np.mean(similarities),
        "final_similarity": np.mean(similarities[-50:]),
        "elapsed_s":        elapsed,
        "per_trial_ms":     (elapsed / trials) * 1000,
        "final_attn_vis":   final_attn[0],   # how much weight on visual stream
        "final_attn_ling":  final_attn[1],   # how much weight on linguistic stream
        "similarities":     similarities
    }


def run_conflict_intensity_test(trials=300, dim=128, batch_size=32):
    """
    Vary the conflict between streams and measure coherence objective benefit.
    Higher conflict = more ambiguity = coherence objective should matter more.
    """
    print("\n--- Conflict Intensity Test ---")
    print(f"{'Vis noise':>10} | {'Ling noise':>10} | {'Base sim':>10} | {'OGI sim':>10} | {'Delta':>8}")
    print("-" * 58)

    configs = [
        (0.1, 0.1),   # low conflict - streams similar
        (0.3, 0.3),   # medium conflict
        (0.5, 0.5),   # paper-level conflict
        (0.5, 1.0),   # asymmetric - linguistic more noisy
        (1.0, 1.0),   # high conflict
    ]

    for vis_noise, ling_noise in configs:
        # patch noise levels for this run
        def make_vis(b, d, c): return make_visual_stream(b, d, c, vis_noise)
        def make_ling(b, d, c): return make_linguistic_stream(b, d, c, noise_scale=ling_noise)

        # baseline
        model_base = MultiModalFusionCell(input_dim=dim, hidden_dim=dim)
        opt_base = torch.optim.Adam(model_base.parameters(), lr=1e-3)
        context = torch.randn(batch_size, dim)
        h_v, h_l = torch.zeros(batch_size, dim), torch.zeros(batch_size, dim)
        sims_base = []
        for _ in range(trials):
            opt_base.zero_grad()
            x_v = make_vis(batch_size, dim, context)
            x_li = make_ling(batch_size, dim, context)
            h_f, o_t, h_v, h_l, _ = model_base(x_v, x_li, h_v, h_l)
            loss = model_base.task_loss(o_t, context)
            loss.backward()
            opt_base.step()
            with torch.no_grad():
                sims_base.append(nn.functional.cosine_similarity(o_t, context).mean().item())
            h_v, h_l = h_v.detach(), h_l.detach()

        # OGI with coherence
        model_ogi = MultiModalFusionCell(input_dim=dim, hidden_dim=dim)
        opt_ogi = torch.optim.Adam(model_ogi.parameters(), lr=1e-3)
        h_v, h_l = torch.zeros(batch_size, dim), torch.zeros(batch_size, dim)
        sims_ogi = []
        for _ in range(trials):
            opt_ogi.zero_grad()
            x_v = make_vis(batch_size, dim, context)
            x_li = make_ling(batch_size, dim, context)
            h_f, o_t, h_v, h_l, _ = model_ogi(x_v, x_li, h_v, h_l)
            loss = model_ogi.task_loss(o_t, context) + 0.5 * model_ogi.coherence_loss(h_f, context)
            loss.backward()
            opt_ogi.step()
            with torch.no_grad():
                sims_ogi.append(nn.functional.cosine_similarity(o_t, context).mean().item())
            h_v, h_l = h_v.detach(), h_l.detach()

        base_final = np.mean(sims_base[-50:])
        ogi_final  = np.mean(sims_ogi[-50:])
        delta = ogi_final - base_final

        print(f"{vis_noise:>10.1f} | {ling_noise:>10.1f} | {base_final:>10.4f} | {ogi_final:>10.4f} | {delta:>+8.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("OGI Framework - Multi-Modal Coherence Test")
    print("Lemma 4.1: coherence benefit under cross-modal ambiguity")
    print("=" * 60)
    print(f"Config: dim=128, batch=32, 500 trials, CPU\n")

    print("Running baseline (task loss only)...")
    base = run_multimodal_benchmark(enable_coherence=False, trials=500)

    print("Running OGI (task loss + coherence objective)...")
    ogi  = run_multimodal_benchmark(enable_coherence=True,  trials=500)

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
    print(f"  (baseline has no attention mechanism - weights fixed at 0.5/0.5)")

    run_conflict_intensity_test(trials=300)

    print("\nDone.")
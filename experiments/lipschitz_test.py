"""
Lipschitz Verification - Theorem 1
Empirical test of the stability bound from Section II-D

Theorem 1 claims:
    ||Phi(c1, et) - Phi(c2, et)||_2 <= (L_A / 2) * ||c1 - c2||_2
    where L_A = ||W_a||_2 * ||W_c||_2

Running on laptop, no GPU
TODO: rerun with larger n_modules and more pairs when GPU available
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')  # no display needed, just saving the plot
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)


class AttentionWeighting(nn.Module):
    """
    Phi(c, et) = softmax(A(c, et))
    A(c, et)   = W_a^T tanh(W_c c + W_e et + b_a)
    Equations 1-2 in the paper
    """
    def __init__(self, dc, de, da, n_modules):
        super().__init__()
        self.W_c = nn.Linear(dc, da, bias=False)
        self.W_e = nn.Linear(de, da, bias=False)
        self.b_a = nn.Parameter(torch.zeros(da))
        self.W_a = nn.Linear(da, n_modules, bias=False)

    def forward(self, c, et):
        a = self.W_a(torch.tanh(self.W_c(c) + self.W_e(et) + self.b_a))
        return torch.softmax(a, dim=-1)

    def lipschitz_bound(self):
        # L_A = ||W_a||_2 * ||W_c||_2
        # softmax contributes factor of 1/2 (see proof)
        # so theoretical bound is L_A / 2
        W_a_norm = torch.linalg.matrix_norm(self.W_a.weight, ord=2).item()
        W_c_norm = torch.linalg.matrix_norm(self.W_c.weight, ord=2).item()
        L_A = W_a_norm * W_c_norm
        return L_A / 2, L_A, W_a_norm, W_c_norm


def run_lipschitz_test(n_pairs=1000, dc=64, de=64, da=64, n_modules=16,
                       perturbation_scales=None):
    """
    For n_pairs random (c1, c2) pairs at varying distances:
    - compute ||Phi(c1) - Phi(c2)||_2  (observed output distance)
    - compute (L_A/2) * ||c1 - c2||_2  (theoretical bound)
    - verify observed <= bound for all pairs
    - compute empirical Lipschitz constant (max ratio)

    If the theorem holds, every point should fall below the bound line.
    """
    if perturbation_scales is None:
        perturbation_scales = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]

    model = AttentionWeighting(dc=dc, de=de, da=da, n_modules=n_modules)
    model.eval()  # no dropout etc., just clean forward passes

    bound, L_A, W_a_norm, W_c_norm = model.lipschitz_bound()

    print(f"Model config: dc={dc}, de={de}, da={da}, n_modules={n_modules}")
    print(f"||W_a||_2 = {W_a_norm:.4f}")
    print(f"||W_c||_2 = {W_c_norm:.4f}")
    print(f"L_A = {L_A:.4f}")
    print(f"Theoretical Lipschitz bound (L_A/2) = {bound:.4f}")
    print()

    # fixed task embedding for all pairs - holds et constant as in the theorem
    et = torch.randn(1, de)

    all_input_dists  = []
    all_output_dists = []
    all_bounds       = []
    violations       = 0

    print(f"{'Scale':>8} | {'Pairs':>6} | {'Max ratio':>10} | {'Violations':>10} | {'Bound holds':>12}")
    print("-" * 60)

    for scale in perturbation_scales:
        input_dists  = []
        output_dists = []

        for _ in range(n_pairs):
            c1 = torch.randn(1, dc)
            # c2 is c1 plus a perturbation at the given scale
            c2 = c1 + torch.randn(1, dc) * scale

            with torch.no_grad():
                phi1 = model(c1, et)
                phi2 = model(c2, et)

            input_dist  = torch.norm(c1 - c2).item()
            output_dist = torch.norm(phi1 - phi2).item()
            theoretical = bound * input_dist

            input_dists.append(input_dist)
            output_dists.append(output_dist)

            all_input_dists.append(input_dist)
            all_output_dists.append(output_dist)
            all_bounds.append(theoretical)

            if output_dist > theoretical + 1e-6:  # small tolerance for float errors
                violations += 1

        ratios = [o / i for o, i in zip(output_dists, input_dists) if i > 1e-8]
        max_ratio = max(ratios)
        scale_violations = sum(1 for o, i in zip(output_dists, input_dists)
                               if o > bound * i + 1e-6)

        print(f"{scale:>8.2f} | {n_pairs:>6} | {max_ratio:>10.4f} | "
              f"{scale_violations:>10} | {'YES' if scale_violations == 0 else 'NO':>12}")

    total_pairs = n_pairs * len(perturbation_scales)
    empirical_lipschitz = max(o / i for o, i in zip(all_output_dists, all_input_dists)
                              if i > 1e-8)

    print()
    print(f"Total pairs tested:        {total_pairs}")
    print(f"Total violations:          {violations}")
    print(f"Theoretical bound (L_A/2): {bound:.4f}")
    print(f"Empirical Lipschitz const: {empirical_lipschitz:.4f}")
    print(f"Bound tightness:           {empirical_lipschitz / bound:.2%} of theoretical max")
    print()

    if violations == 0:
        print("THEOREM 1 VERIFIED: bound holds across all tested pairs.")
    else:
        print(f"WARNING: {violations} violations detected - check float tolerance or model config.")

    # save a plot - input distance vs output distance with bound line overlaid
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(all_input_dists, all_output_dists, alpha=0.3, s=8,
               label='Observed', color='steelblue')

    x_line = np.linspace(0, max(all_input_dists), 100)
    ax.plot(x_line, bound * x_line, 'r-', linewidth=2,
            label=f'Theoretical bound (L_A/2 = {bound:.3f})')

    ax.set_xlabel('Input distance ||c1 - c2||_2')
    ax.set_ylabel('Output distance ||Phi(c1) - Phi(c2)||_2')
    ax.set_title('Theorem 1: Lipschitz Stability Verification')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('lipschitz_verification.png', dpi=150)
    print("Plot saved: lipschitz_verification.png")

    return {
        "bound": bound,
        "empirical_lipschitz": empirical_lipschitz,
        "violations": violations,
        "total_pairs": total_pairs,
        "tightness": empirical_lipschitz / bound
    }


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("OGI Framework - Theorem 1 Lipschitz Verification")
    print("Empirical test of Section II-D stability bound")
    print("=" * 60)
    print()

    results = run_lipschitz_test(
        n_pairs=1000,
        dc=64,
        de=64,
        da=64,
        n_modules=16,
        perturbation_scales=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
    )

    print()
    print("These results go in Section VI-C (Lipschitz Verification).")
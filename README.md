# ogi-math-foundations

**Formal Mathematical Foundations for the Open General Intelligence Framework**

A companion paper and reference implementation providing the mathematical rigor that the original OGI architectural specification required but did not include.

---

## What This Is

The [Open General Intelligence (OGI) framework](https://arxiv.org/abs/2411.15832) (Dollinger & Singleton, 2024) proposed a modular cognitive architecture for AGI. The vision was compelling. The math wasn't there yet.

This repository contains the paper and benchmark code that put a formal floor under that vision:

- **Stability proofs** for the attention-based weighting system (Lipschitz bounds)
- **Complexity analysis** of the fabric interconnect protocol and the proof that it's O(n²), which breaks at scale
- **Top-K Gating** (Theorem 3): the biologically-motivated fix that reduces complexity from O(n²dₘ) to O(k²dₘ + n(d꜀d_a))
- **Theorem 4** with two supporting lemmas: GRU-based fusion preserves semantic relationships across modalities under explicit sufficient conditions
- **Simulation results**: temporal stability validated across 200 processing cycles at five noise levels (σ ∈ {0.0, 0.25, 0.50, 0.75, 1.0}), with honest reporting of what the coherence objective does and doesn't buy you in single-modality conditions

This is step two of a three-step sequence. [1] was the vision. This is the formal specification. The implementation paper comes next.

---

## Paper

> Singleton, M. (2025). *Addressing Mathematical Rigor in the Open General Intelligence Framework: A Critical Analysis and Formal Implementation.* Zenodo. [DOI: 10.5281/zenodo.19135205]

Preprint available on Zenodo. arXiv submission pending endorsement in cs.AI.

---

## Key Results

| Metric | Result |
|--------|--------|
| Temporal stability at σ = 0.5 (Gaussian noise) | 0.9940 cosine similarity, 181/200 cycles stable |
| Temporal stability at σ = 1.0 | 0.9724 cosine similarity, 173/200 cycles stable |
| Coherence Tax (compute overhead) | +73% per-trial vs. baseline GRU |
| Complexity reduction via Top-K Gating | O(n²dₘ) → O(k²dₘ + n(d꜀d_a)) |

The stability test is the strongest empirical result. The -0.3% accuracy delta between baseline and OGI on single-modality reconstruction is theoretically expected and explained in Section VI-B of the paper — the coherence objective's benefit is conditional on genuine cross-modal ambiguity.

---

## Benchmark Code

```bash
pip install torch numpy
python ogi_benchmark.py
```

Runs on CPU. No GPU required. Expect ~2 seconds total on a modern laptop.

**What it tests:**
- Theorem 4 validation: GRU fusion with coherence objective vs. task-loss-only baseline
- Lemma 4.2 validation: temporal stability across 200 cycles at five noise levels
- Coherence Tax measurement: per-trial compute overhead of the MINE critic

**Reproducibility:** `torch.manual_seed(42)`, `dim=128`, `batch=32`, `500 trials`. Results in the paper were generated on CPU (Intel, Windows 11).

---

## Repository Structure

```
ogi-math-foundations/
├── README.md
├── CITATION.cff
├── LICENSE
├── ogi_benchmark.py        # Benchmark code (Theorem 4 + Lemma 4.2 validation)
└── paper/
    └── OGI_Math_Foundations.pdf
```

---

## Citation

```bibtex
@misc{singleton2025ogi,
  author    = {Singleton, Michael},
  title     = {Addressing Mathematical Rigor in the Open General Intelligence 
               Framework: A Critical Analysis and Formal Implementation},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19135205}
}
```

---

## Related Work

- **OGI Framework (original):** Dollinger, D.A. & Singleton, M. (2024). [arXiv:2411.15832](https://arxiv.org/abs/2411.15832)
- **Attention mechanism:** Vaswani et al. (2017). *Attention Is All You Need.* NeurIPS.
- **MINE:** Belghazi et al. (2018). *Mutual Information Neural Estimation.* ICML.
- **Sparse MoE:** Shazeer et al. (2017). *Outrageously Large Neural Networks.* ICLR.

---

## What Comes Next

The outstanding work, in order of priority:

1. **Multi-modal coherence experiment** — two genuinely distinct input streams with conflicting low-level features. This is where Lemma 4.1 gets its real test.
2. **Lipschitz verification** — empirical measurement of ‖Φ(c₁, eᵗ) − Φ(c₂, eᵗ)‖₂ against predicted bounds.
3. **Training algorithm** — formal specification of a complete training procedure for the full OGI system. The biggest gap between this paper and a running implementation.
4. **Hardware-scale Coherence Tax** — characterizing the 73% CPU overhead at deployment-relevant scales with GPU and sparse accelerators.

If you're working on any of these, reach out.

---

## Author

**Michael Singleton**
Iconsoft LLC, Mechanicsville, Virginia
[mike@mikesingleton.info](mailto:mike@mikesingleton.info)

IEEE member. Applied Mathematics (MIT). Prior work: [OGI Framework](https://arxiv.org/abs/2411.15832) (SysCon 2024).

---

## License

Paper: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
Code: [MIT License](LICENSE)

# ogi-math-foundations

Formal mathematical foundations for the OGI cognitive architecture framework — stability proofs, complexity analysis, Top-K Gating, and GRU-based semantic fusion. Companion to [arXiv:2411.15832](https://arxiv.org/abs/2411.15832).

---

## What This Is

The [Open General Intelligence (OGI) framework](https://arxiv.org/abs/2411.15832) (Dollinger & Singleton, 2024) proposed a modular cognitive architecture for AGI. The vision was sound. The mathematical floor wasn't there.

This repository provides that floor: formal proofs, empirical validation, and benchmark code for the four core results the original paper required but did not include.

**This is step two of a three-step sequence.** Step one was the architectural vision. This is the formal specification. The implementation paper comes next.

**The navi-SAD instrument** (Spence, Project-Navi) is integrated here as an impartial observer for the OGI Executive Gater — measuring spectral attention divergence as a dynamical systems probe during inference. See [`instruments/`](instruments/) and [navi-SAD](https://github.com/Project-Navi/navi-SAD).

---

## Key Results

| Result | Metric | Value |
|---|---|---|
| Temporal stability at σ = 0.5 | Cosine similarity / stable cycles | 0.9940 / 181 of 200 |
| Temporal stability at σ = 1.0 | Cosine similarity / stable cycles | 0.9724 / 173 of 200 |
| Complexity reduction via Top-K Gating | Before → After | O(n²dₘ) → O(k²dₘ + n(d꜀d_a)) |
| Coherence Tax (compute overhead) | Per-trial vs. baseline GRU | +73% |

The stability result is the strongest empirical claim. The −0.3% accuracy delta between baseline and OGI on single-modality reconstruction is theoretically expected: the coherence objective's benefit is conditional on genuine cross-modal ambiguity, explained in Section VI-B of the paper.

---

## Theorems

| | Claim | Status |
|---|---|---|
| **Theorem 1** | Lipschitz stability bounds on the attention-based weighting system | Proved |
| **Theorem 3** | Top-K Gating reduces fabric interconnect complexity from O(n²dₘ) | Proved |
| **Lemma 4.1** | Mutual information-driven routing preserves cross-modal relationships | Proved |
| **Lemma 4.2** | GRU mapping is contractive under explicit sufficient conditions | Proved |
| **Theorem 4** | GRU-based fusion preserves semantic relationships under Lemmas 4.1–4.2 | Proved |

Validated across 200 processing cycles at five noise levels (σ ∈ {0.0, 0.25, 0.50, 0.75, 1.0}).

---

## Paper

> Singleton, M. (2025). *Addressing Mathematical Rigor in the Open General Intelligence Framework: A Critical Analysis and Formal Implementation.* Zenodo. DOI: [10.5281/zenodo.19135205](https://doi.org/10.5281/zenodo.19135205)

arXiv submission pending endorsement in cs.AI.

---

## Installation

```bash
pip install -r requirements.txt
```

Runs on CPU. No GPU required. Expect ~2 seconds on a modern laptop.

---

## Benchmark

```bash
python experiments/ogi_benchmark.py
```

**What it tests:**

- Theorem 4 validation: GRU fusion with coherence objective vs. task-loss-only baseline
- Lemma 4.2 validation: temporal stability across 200 cycles at five noise levels
- Coherence Tax: per-trial compute overhead of the MINE critic

**Reproducibility:** `torch.manual_seed(42)`, `dim=128`, `batch=32`, `500 trials`. Results generated on CPU (Intel, Windows 11).

---

## Repository Structure

```
ogi-math-foundations/
├── README.md
├── CITATION.cff
├── CONTRIBUTING.md
├── ROADMAP.md
├── LICENSE
├── requirements.txt
├── Makefile
├── paper/
│   └── OGI_Math_Foundations.pdf
├── experiments/
│   ├── ogi_benchmark.py          # Theorem 4 + Lemma 4.2 validation (primary)
│   ├── lipschitz_test.py         # Theorem 1 empirical verification
│   ├── multimodal_test.py        # Cross-modal coherence experiment
│   ├── clip_coherence_test.py    # CLIP-based coherence objective test
│   └── semantic_incompleteness.py
├── instruments/
│   └── og_navisad_v3.py          # navi-SAD integration (Executive Gater observer)
└── notebooks/
    ├── OGI_Demo_Colab.ipynb
    └── OGI_Demo_Azure.ipynb
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
- **navi-SAD (instrument):** Spence, N. Project-Navi. [github.com/Project-Navi/navi-SAD](https://github.com/Project-Navi/navi-SAD)
- **Attention mechanism:** Vaswani et al. (2017). *Attention Is All You Need.* NeurIPS.
- **MINE:** Belghazi et al. (2018). *Mutual Information Neural Estimation.* ICML.
- **Sparse MoE:** Shazeer et al. (2017). *Outrageously Large Neural Networks.* ICLR.

---

## What Comes Next

See [ROADMAP.md](ROADMAP.md) for the full research plan. Immediate priorities:

1. **Multi-modal coherence experiment** — two genuinely distinct input streams with conflicting low-level features. The real test for Lemma 4.1.
2. **Lipschitz verification** — empirical measurement of ‖Φ(c₁, eᵗ) − Φ(c₂, eᵗ)‖₂ against predicted bounds.
3. **OGI Fabric experiment** — inter-module message passing with navi-SAD as impartial observer.
4. **Training algorithm** — formal specification of a complete training procedure for the full OGI system.

If you are working on any of these, reach out.

---

## Author

**Michael Singleton**  
Iconsoft LLC | Richmond, Virginia  
[mike.singleton@iconsoft.us](mailto:mike.singleton@iconsoft.us)

IEEE member. M.S. Applied Mathematics (MIT). Prior work: [OGI Framework](https://arxiv.org/abs/2411.15832) (IEEE SysCon 2024).

---

## License

Paper: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)  
Code: [MIT License](LICENSE)

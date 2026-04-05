# Roadmap

Outstanding work in order of priority. This is a research repository — timelines are estimates, not commitments.

---

## Immediate (next experiment)

### OGI Fabric — Inter-module message passing
The next planned validation target. navi-SAD integrated as impartial observer on the Executive Gater. Goal: measure spectral attention divergence during cross-module routing and characterize attractor geometry under Top-K Gating vs. full fabric.

**Dependency:** navi-SAD Gate 3 (synthetic HMM benchmark). See [Project-Navi/navi-SAD](https://github.com/Project-Navi/navi-SAD).

---

## Near-term

### Multi-modal coherence experiment
Two genuinely distinct input streams with conflicting low-level features. This is where Lemma 4.1 gets its real test — the single-modality benchmark cannot validate the cross-modal claim.

**Expected outcome:** Positive coherence delta (vs. the −0.3% seen in single-modality). If not, the coherence objective needs to be scoped more narrowly.

### Lipschitz verification
Empirical measurement of ‖Φ(c₁, eᵗ) − Φ(c₂, eᵗ)‖₂ against the bounds proved in Theorem 1. Current lipschitz_test.py is a scaffold; this experiment closes the loop between the formal proof and measured behavior.

---

## Medium-term

### Hardware-scale Coherence Tax
The 73% CPU overhead figure was measured at development scale. Characterizing that overhead at deployment-relevant scales (GPU, sparse accelerators, larger batch sizes) is necessary before any production claim about the coherence objective is credible.

### Training algorithm
Formal specification of a complete training procedure for the full OGI system. Currently the biggest gap between this paper and a running implementation. The math is there; the algorithm that uses it isn't written.

---

## Long-term

### arXiv endorsement (cs.AI)
Submission pending. If you are an arXiv endorser in cs.AI and are willing to review the paper, reach out.

### Implementation paper
Step three of the three-step sequence. Vision (arXiv:2411.15832) → formal specification (this repo) → implementation.

---

## Not Planned

- PyPI packaging (this is a research benchmark, not a library)
- GPU-required experiments (CPU reproducibility is a deliberate constraint)
- Baselines against architectures other than OGI until the OGI implementation is complete

---

*Last updated: 2025*

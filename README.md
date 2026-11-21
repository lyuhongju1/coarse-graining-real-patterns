# Simulation and analysis code

Code and stored results for the paper "Which Coarse-Grainings Are Real Patterns? The Limits of
Informational Individuation". Every figure, table entry and quoted number in the paper is produced by
the scripts here.

## Running

    pip install -r requirements.txt
    sh run_all.sh

Tested with Python 3.13. All scripts use fixed random seeds, write their results to a JSON file in this
folder and take no input other than the command-line arguments shown in `run_all.sh`. The full run takes
roughly ten minutes.

## Model

One-dimensional Ising ring, N = 4096 sites, J = 1, T = 1.6 unless stated, blocks of b = 16 sites.

- **Glauber dynamics**: single-spin flips with probability 1/(1 + exp(dE/T)), even and odd sites updated
  alternately; 300 sweeps discarded, 4000 recorded.
- **Kawasaki dynamics**: exchange of neighbouring antiparallel spins with the same acceptance rule;
  the chain is first equilibrated for 300 Glauber sweeps, then run for 16,300 exchange sweeps, with the
  first 300 discarded and every fourth configuration recorded.

Fields: `A` (block mean, offset 0), `B_k` (block mean, offset k), `Tri` (triangular weights 1..8,8..1),
`Gau` (Gaussian weights, standard deviation 4 sites, 25 taps), `W` (density of domain walls per block).
For entropy estimates the kernel fields are quantized to 17 levels.

## Which script produces what

| Item in the paper | Script | Output |
|---|---|---|
| Spin correlations at T = 1.6 | `run_model.py` | `results.json` (`corr`) |
| Figure 1 (schematic) | `make_figures.py` | `figs/fig_schematic.pdf` |
| Figure 2, left; Table 1 "R2 from A" for B_1, B_8 | `run_model.py` | `results.json` (`derivability`) |
| Figure 2, right; Table 1 "Entropy reduced"; joint reductions in the text | `joint_entropy.py` | `joint.json` |
| Residual entropies of B_8 and the Gaussian field; single-spin conditional entropies; wall-density footnote | `entropy.py` | `entropy.json` |
| Table 1, Glauber one-step R2 and autonomy for A and B_k | `run_model.py` | `results.json` (`autonomy`, `autA_micro`, `autB8_micro`) |
| Table 1, Glauber rows for the kernel fields | `kernel.py` | `kernel.json` |
| Predictive information in bits (three-block conditioning) | `selfent.py` | `selfent.json` |
| Robustness across seeds, block sizes and temperatures; cubic versus quadratic fits | `robust.py` | `robust.json` |
| Two-dimensional check (footnote) | `sim2d.py` | `sim2d.json` |
| Kawasaki self-prediction at 4, 32 and 128 sweeps; Table 1, Kawasaki rows | `kawasaki.py` | `kawasaki_lag{1,8,32}_seed*.json` |
| Magnetization versus wall density at T = 0.8 | `relevance.py 0.8` | `relevance_T0.8.json` |
| Figure 3 (predictability of Fourier modes) | `ib.py`, `make_figures.py` | `ib.json`, `figs/fig_modes.pdf` |
| Exact comparison of joint and single-site information gains on small rings | `exact_check.py` | printed |

## Notes on the entropy measures

- `joint_entropy.py` computes the reduction in the length of description of a configuration when a second
  field is added to `A`, which equals the entropy of that field given `A`. Given the whole of `A`, `B_k` is
  fixed by the sums over the first k sites of each block, so the estimate is made from those partial sums
  with a truncated chain rule. This is the quantity reported in Table 1 and Figure 2.
- `entropy.py` computes single-spin conditional entropies (one spin given the block values covering it).
  These are reported in the text as statements about individual spins. They are not the joint reduction;
  `exact_check.py` shows by full enumeration that the two can differ in either direction.
- All entropy figures are equal-time values.
- The kernel fields are quantized to 17 levels for the entropy estimates; no finer quantization was tested.

## Licence

MIT; see `LICENSE`.

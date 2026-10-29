# Generalization of electromagnetic surrogate models — data and trained models

Test datasets and trained neural networks for

> J. Kim, M. Park, C. Kang, H. Suk, and H. Chung,
> *From random patterns to nanophotonic structures: Generalization of electromagnetic surrogate models* (2026).

The networks were trained on random freeform dielectric patterns and evaluated, without retraining, on the structures below.
All fields are two-dimensional TE solutions computed with a finite-difference frequency-domain (FDFD) solver.

## Contents

| Location | What |
|---|---|
| this repository | `code/` (network definitions and evaluation script), `models/` (trained weights), `requirements.txt` |
| [Releases](../../releases) | `data/` test datasets (HDF5, each file < 2 GB) and `SHA256SUMS.txt` |

## Quick start

```bash
pip install -r requirements.txt
# download the HDF5 files from the Releases page into data/
python code/evaluate.py --model models/dilresnet.pt --data data/manually_defined_patterns.h5
# DilResNet on data/manually_defined_patterns.h5: 2641 samples, mean e = 1.946 %, median e = 0.796 %
```

In Python:

```python
import h5py, sys
sys.path.append("code")
from networks import load_model
from evaluate import predict, relative_error

model, ck = load_model("models/dilresnet.pt")
with h5py.File("data/manually_defined_patterns.h5") as f:
    eps_r, hz = f["eps_r"][:16], f["hz"][:16]
hz_pred = predict(model, ck, eps_r)          # complex (16, 128, 64)
print(relative_error(hz_pred, hz))           # per-sample relative error
```

Results differ from the paper only at the level of floating-point precision
(the paper's evaluation used mixed precision).

## Test datasets (`data/`)

| File | Samples | Used in |
|---|---:|---|
| `heldout_combinations_part{1..4}of4.h5` | 70,012 | Held-out geometry–permittivity combinations (Tables 1 and 2, Figures 4 and 5) |
| `aperiodic_bar_arrays.h5` | 13,900 | Aperiodic bar arrays (Tables 1 and 2, Figure 3) |
| `manually_defined_patterns.h5` | 2,641 | Manually defined patterns (Tables 1 and 2, Figure 3) |
| `inverse_designed_metalenses.h5` | 6,950 | Metalens-derived structures (Tables 1–3, Figures 3 and 6) |
| `inverse_designed_binary_structures.h5` | 6,950 | Inverse-designed binary structures (Tables 1 and 2, Figure 3) |
| `two_material_freeform_patterns.h5` | 5,000 | Two-material freeform patterns (Table 1, Figure 3) |
| `structural_scale_sweep.h5` | 15,456 | Structural-scale sweep (Figure 7) |
| `wavelength_fixed_grid_part{1,2}of2.h5` | 19,600 | Wavelength sweep, fixed 10 nm grid (Figures S1 and S2) |
| `wavelength_scaled_grid.h5` | 15,680 | Wavelength sweep, grid scaled as λ/50 (Figures S1 and S2) |

Each file is HDF5 with

| Dataset | Shape / type | Meaning |
|---|---|---|
| `eps_r` | (N, 128, 64) float32 | relative permittivity (structure) |
| `hz` | (N, 128, 64) complex64 | complex magnetic field H<sub>z</sub> (FDFD reference) |
| `eps_design` | (N,) float32 | permittivity assigned to the design material (`eps_design_a`, `eps_design_b` for two-material freeform patterns) |
| `geometry_id` | (N,) int32 | geometry index within the set (one geometry is evaluated at several permittivities) |
| `generator_spatial_frequency_per_um` | (N,) float32 | structural-scale sweep only |
| `wavelength_nm`, `grid_spacing_nm` | (N,) float32 | wavelength sweeps only |

Simulation setting (also stored as file attributes): vacuum wavelength 500 nm and grid spacing 10 nm unless stated;
128 × 64 cells (1280 × 640 nm); array index `[n, y, x]` with `y = 0` at the bottom; rows 0–7 substrate (ε<sub>r</sub> = 2.1),
rows 8–20 design layer, rows 21–127 air; plane-wave illumination from the substrate side; lossless, nondispersive media;
time convention exp(+jωt).
In `inverse_designed_metalenses.h5` (metalens-derived structures), the non-material cells of the design layer have ε<sub>r</sub> = 2.1 instead of air.

## Trained models (`models/`)

| File | Model |
|---|---|
| `fno.pt` | Fourier neural operator (FNO) |
| `ffno.pt` | Factorized FNO (F-FNO) |
| `unet.pt` | U-Net |
| `dilresnet.pt` | Dilated residual network (DilResNet) |
| `fno_wave_residual.pt`, `unet_wave_residual.pt` | FNO and U-Net trained with an additional normalized-output residual regularizer (Figure S14, Table S6) |

Each `.pt` file is a PyTorch dictionary with `architecture`, `hyperparameters`, `state_dict`, `output_mean`, and `output_std`;
`code/networks.py` builds the network from it (`load_model`). `model_cards.json` lists the hyperparameters, parameter counts, and the input/output convention used by `code/evaluate.py`:

- **Input** (batch, 3, 128, 64): normalized permittivity `clip((eps_r − lo) / (hi − lo), 0, 1)` with
  `lo = min(2.1, min eps_r)` and `hi = max(16.0, max eps_r)`; x coordinate `linspace(−1, 1, 64)`; y coordinate `linspace(−1, 1, 128)`.
- **Output** (batch, 2, 128, 64): normalized Re H<sub>z</sub> and Im H<sub>z</sub>;
  `H_z = (out[:, 0] · output_std[0] + output_mean[0]) + j (out[:, 1] · output_std[1] + output_mean[1])`.

The prediction error reported in the paper is the relative error of the complex field,
e = ‖Ĥ<sub>z</sub> − H<sub>z</sub>‖<sub>2</sub> / ‖H<sub>z</sub>‖<sub>2</sub>, over the full 128 × 64 domain.

## Integrity

`SHA256SUMS.txt` lists checksums of the data files and model weights.

## License

Data and model weights are released under the Creative Commons Attribution 4.0 International License (CC BY 4.0).
Please cite the paper above when using them.

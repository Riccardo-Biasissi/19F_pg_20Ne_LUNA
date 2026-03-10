# 19F(p,γ)20Ne and 19F(p,α)16O — LUNA Experiment

This repository contains the data analysis and simulation code for the measurement of the **¹⁹F + p → ²⁰Ne** nuclear reactions at the [LUNA](https://luna.lngs.infn.it/) (Laboratory for Underground Nuclear Astrophysics) underground accelerator facility at Gran Sasso National Laboratory (LNGS), Italy.

The two reaction channels of interest are:
- **¹⁹F(p,γ)²⁰Ne** — proton-capture with gamma emission
- **¹⁹F(p,α)¹⁶O** — proton-capture with alpha emission

These reactions are relevant to stellar nucleosynthesis, in particular to the CNO and NeNa cycles in AGB stars and novae. The experiment aims to measure the astrophysical S-factor at energies directly relevant to stellar conditions.

---

## Repository Structure

```
19F_pg_20Ne_LUNA/
├── Calibration/            # Detector energy and efficiency calibration
├── Computations/           # Target depth-profile analysis and R-matrix fits
│   └── Profile/            # Per-target implantation profile notebooks
├── Data/                   # Raw ROOT files, processed data, stopping powers
│   ├── CURRENT/            # Live data acquisition files
│   ├── PROCESSED/          # Calibrated and processed outputs
│   ├── ROOT/               # ROOT format detector data
│   ├── SIMULATIONS/        # Geant4 / SRIM simulation outputs
│   └── STOPPING/           # SRIM stopping power tables
├── Notes/                  # Experimental notes and references
├── Simulations/            # Yield and cross-section simulations
├── Tests/                  # Validation against published data
│   └── Zhang_2022_pag_yields/
├── Yield_longruns/         # Analysis of long-duration beam runs
├── Yield_scans/            # Energy-scan yield measurements
├── Lost/                   # Archived / obsolete code (kept for reference)
├── compute_yield.py        # Utility script: yield per projectile
├── plotter.ipynb           # Central analysis and publication-quality plots
└── 19F+p_400kV.xlsx        # Master run log and metadata (400 keV campaign)
```

---

## Experimental Setup

- **Accelerator:** LUNA 400 kV electrostatic accelerator
- **Beam:** Proton beam at energies ~100–400 keV (lab frame)
- **Targets:**
  - ¹⁹F implanted into metallic backings (Fe, Ta)
  - ¹⁹F compounds (CaF₂, LiF)
  - ¹⁹F fluorinated on metallic backings (Ta)
- **Detectors:**
  - BGO (Bismuth Germanate) — high-efficiency summing detector in coincidence mode (6 crystals)
  - HPGe (High-Purity Germanium) — high-resolution gamma spectroscopy
- **Coincidence window:** 100–1000 ns
- **Operating temperature:** ~80 °C (relevant for Doppler broadening corrections)

---

## Analysis Workflow

### 1. Detector Calibration (`Calibration/`)

Energy and efficiency calibration of HPGe and BGO detectors using radioactive sources with known gamma-ray lines (511 keV, 5617 keV, 6128 keV, 11660 keV).

| Notebook | Purpose |
|---|---|
| `calibrator_hpge.ipynb` | HPGe energy calibration |
| `calibrator_sources.ipynb` | Source-based calibration |
| `calibrator_scans.ipynb` | Calibration for scan data |
| `calibrator_longruns.ipynb` | Calibration for long runs |
| `BGOsumming.py` | Applies calibration to BGO ROOT files, handles coincidence |

Per-run calibration parameters are stored in `Calibration/Params/calibration_run*.txt`.

### 2. Target Profile Analysis (`Computations/Profile/`)

The ¹⁹F implantation depth profile in each target is determined by fitting the yield-vs-energy profile using a combination of:
- SRIM-based stopping power calculations
- Skew-Gaussian or Gaussian profile functions
- Doppler broadening corrections

| Script | Purpose |
|---|---|
| `profile_fit_240.py` | Profile fit at 240 keV target scans |
| `profile_fit_340.py` | Profile fit at 340 keV target scans |
| `SRIM.py` | SRIM output parser (dE/dx, range, straggling) |
| `pyazr/` | Azure2 R-matrix code interface |

Target types analysed: `IMP_LFE#` (Fe backing), `IMP_LTA#` (Ta backing), `SUDF#`, `EVA_AT#`.

### 3. Energy Scan Yields (`Yield_scans/`)

Reaction yield is measured as a function of proton energy to map out resonance structures.

| File | Purpose |
|---|---|
| `yield_pg_beamtime5.ipynb` | Main scan yield extraction (beam time 5) |
| `Profiler.ipynb` | Beam energy profile analysis |
| `Results/Yield_*.csv` | Extracted yield tables per target |

### 4. Long-Run Yields and S-factor (`Yield_longruns/`)

Long-duration runs at fixed energies maximise statistical precision for the absolute cross-section measurement.

| File | Purpose |
|---|---|
| `yield_pa.ipynb` | Proton-alpha channel yield extraction |
| `s_factor_computation.ipynb` | Convert yield to astrophysical S-factor |
| `matrix_gif.ipynb` | Multi-dimensional data visualisation |
| `utils/*.extrap` | R-matrix extrapolation files |
| `Yield_6130_total.csv` | Yield for the 6130 keV gamma line |
| `JUNA_S_factor.csv` | S-factor comparison with JUNA experiment |

The astrophysical S-factor is defined as:

```
S(E) = E · σ(E) · exp(2πη)
```

where `η` is the Sommerfeld parameter and `E` is the center-of-mass energy.

### 5. Simulations (`Simulations/`)

| File | Purpose |
|---|---|
| `Calculate_pag.ipynb` | Simulate ¹⁹F(p,γ) and ¹⁹F(p,α) yields |
| `Convert_pg.ipynb` | Convert proton-gamma simulation data |
| `SRIM.py` | SRIM stopping power parser |
| `stopping/` | SRIM output tables for various ion-target combinations |

### 6. Validation (`Tests/`)

The `Zhang_2022_pag_yields/` folder contains cross-checks of the LUNA analysis against published results from Zhang et al. 2022 (JUNA facility), ensuring consistency in resonance parameters and energy calibration.

### 7. Final Plots (`plotter.ipynb`)

The `plotter.ipynb` notebook integrates all analysis outputs and produces the final publication-quality figures summarising cross-sections, S-factors, and comparisons with other experiments.

---

## Key Scripts

### `compute_yield.py`

Utility to compute the thin-target reaction yield per projectile:

```bash
python compute_yield.py <energy_keV> <cross_section_barn>
```

- Loads SRIM stopping power tables for H in F and H in Ta
- Computes effective stopping power for compound targets
- Returns dimensionless yield per projectile

### `Calibration/BGOsumming.py`

Processes BGO detector ROOT files:
- Applies per-run energy calibration
- Handles BGO–HPGe coincidence summing
- Configurable coincidence window (default 100 ns)

### `Computations/Profile/SRIM.py` / `Simulations/SRIM.py`

SRIM output parser providing interpolation of:
- Electronic and nuclear stopping power `dE/dx`
- Projected range
- Longitudinal and lateral straggling

---

## Data Files

| File/Directory | Format | Contents |
|---|---|---|
| `Data/STOPPING/` | SRIM text tables | Stopping powers for H in F, Fe, Ta, Ca, Cr, … |
| `Calibration/Params/` | Text | Per-run HPGe/BGO calibration parameters |
| `Yield_scans/Results/*.csv` | CSV | Energy-resolved yield per target |
| `Yield_longruns/Yield_6130_total.csv` | CSV | Long-run yield for 6130 keV line |
| `Yield_longruns/JUNA_S_factor.csv` | CSV | Reference S-factor from JUNA |
| `Yield_longruns/utils/*.extrap` | ASCII | R-matrix extrapolation tables |
| `19F+p_400kV.xlsx` | Excel | Master run log for 400 keV campaign |

---

## Physical Constants and Parameters

| Quantity | Value |
|---|---|
| Proton mass | 1.0078250 amu |
| ¹⁹F mass | 18.998403 amu |
| ¹⁹F nuclear charge | Z = 9 |
| Beam energies (lab) | ~300–400 keV |
| CM energy range | ~50–300 keV |
| BGO coincidence window | 100–1000 ns |

---

## Dependencies

The analysis notebooks and scripts use the following Python packages:

- `numpy`, `scipy` — numerical computing and interpolation
- `matplotlib` — plotting
- `pandas` — data handling
- `lmfit` — parameter fitting (profile analysis)
- `uproot` or `ROOT` — reading ROOT format detector data
- `jupyter` — interactive notebooks

---

## References

- LUNA collaboration: https://luna.lngs.infn.it/
- Zhang et al. 2022 — ¹⁹F(p,α)¹⁶O measurement at JUNA
- SRIM stopping power code: http://www.srim.org/
- Azure2 R-matrix code: https://azure.nd.edu/

---

## Notes

- The `Lost/` directory contains archived code kept for historical reference; it is not part of the active analysis pipeline.
- Runs in the energy region 216–260 keV are excluded from certain S-factor analyses due to resonance interference; see `s_factor_computation.ipynb` for details.
- All yields are corrected for beam current integration (charge in μC), dead time, and detector efficiency.

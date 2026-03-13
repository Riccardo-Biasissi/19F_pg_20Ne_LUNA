# =============================================================================
#  fit_parallel.py  —  versione parallelizzata con multiprocessing.Pool
#
#  Strategia:
#    1. straggled_profile  →  ogni punto x[idx] viene calcolato in parallelo
#                             (questo è il vero bottleneck)
#    2. loop target/scan   →  ogni combinazione (target, scan) gira in parallelo
#                             (utile quando hai molti target)
#
#  Note importanti:
#    • lmfit.Parameters non è picklable → viene convertito in dict prima di
#      essere passato ai worker
#    • Non si annidano mai due Pool → la parallelizzazione su target/scan
#      chiama straggled_profile seriale internamente, oppure si sceglie solo
#      uno dei due livelli (vedi FLAG USE_SCAN_LEVEL_PARALLEL sotto)
#    • emcee lancia già thread interni → il blocco MCMC resta seriale
#    • _skg_cache è locale a ogni worker → nessun problema di condivisione
# =============================================================================

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # backend non-interattivo, obbligatorio nei worker
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
import os
import re
from functools import partial

from scipy import integrate
from lmfit import Parameters, minimize

from scipy.special import gamma
from scipy.special import erf
from scipy.stats import skewnorm

import emcee
import corner

# ── multiprocessing ──────────────────────────────────────────────────────────
from multiprocessing import Pool
import multiprocessing as mp

# =============================================================================
#  FLAG DI CONFIGURAZIONE PARALLELIZZAZIONE
# =============================================================================
# Scegli UNO dei due livelli (non entrambi True contemporaneamente):
#   PROFILE_LEVEL_PARALLEL = True  → parallelizza i punti interni di
#                                     straggled_profile (consigliato se hai
#                                     pochi target/scan ma molti punti x)
#   SCAN_LEVEL_PARALLEL    = True  → parallelizza il loop target/scan
#                                     (consigliato se hai molti target/scan)
PROFILE_LEVEL_PARALLEL = True   # parallelizza straggled_profile
SCAN_LEVEL_PARALLEL    = False  # parallelizza loop target/scan
N_WORKERS = 16                  # None → usa tutti i core disponibili

# =============================================================================
#  IMPOSTAZIONI MCMC
# =============================================================================
#  fit_parallel.py  —  versione parallelizzata con multiprocessing.Pool
#
#  Strategia:
#    1. straggled_profile  →  ogni punto x[idx] viene calcolato in parallelo
#                             (questo è il vero bottleneck)
#    2. loop target/scan   →  ogni combinazione (target, scan) gira in parallelo
#                             (utile quando hai molti target)
#
#  Note importanti:
#    • lmfit.Parameters non è picklable → viene convertito in dict prima di
#      essere passato ai worker
#    • Non si annidano mai due Pool → la parallelizzazione su target/scan
#      chiama straggled_profile seriale internamente, oppure si sceglie solo
#      uno dei due livelli (vedi FLAG USE_SCAN_LEVEL_PARALLEL sotto)
#    • emcee lancia già thread interni → il blocco MCMC resta seriale
#    • _skg_cache è locale a ogni worker → nessun problema di condivisione
# =============================================================================

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # backend non-interattivo, obbligatorio nei worker
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
import os
import re
from functools import partial

from scipy import integrate
from lmfit import Parameters, minimize

from scipy.special import gamma
from scipy.special import erf
from scipy.stats import skewnorm

import emcee
import corner

# ── multiprocessing ──────────────────────────────────────────────────────────
from multiprocessing import Pool
import multiprocessing as mp

# =============================================================================
#  FLAG DI CONFIGURAZIONE PARALLELIZZAZIONE
# =============================================================================
# Scegli UNO dei due livelli (non entrambi True contemporaneamente):
#   PROFILE_LEVEL_PARALLEL = True  → parallelizza i punti interni di
#                                     straggled_profile (consigliato se hai
#                                     pochi target/scan ma molti punti x)
#   SCAN_LEVEL_PARALLEL    = True  → parallelizza il loop target/scan
#                                     (consigliato se hai molti target/scan)
PROFILE_LEVEL_PARALLEL = True   # parallelizza straggled_profile
SCAN_LEVEL_PARALLEL    = False  # parallelizza loop target/scan
N_WORKERS = 20                  # None → usa tutti i core disponibili

# =============================================================================
#  IMPOSTAZIONI MCMC
# =============================================================================
RUN_MCMC      = True
MCMC_NWALKERS = 16
MCMC_BURN     = 100
MCMC_STEPS    = 1000
MCMC_THIN     = 5
ENERGY_TAG    = "340"

# =============================================================================
#  COSTANTI
# =============================================================================
k           = 8.617e-5
eff_far     = 0.0044
eff_close   = 0.689
q_e         = 1.602176634e-19

# =============================================================================
#  DATI E STOPPING
# =============================================================================
from pyazr import azure2
from SRIM import SRIM

data     = np.loadtxt("Computations/Profile/utils/all_imp.extrap", usecols=(0, 3))
energies = data[:, 0]
extrap   = data[:, 1]

dopp = np.sqrt(2 * 1.007 / 19 * 0.250 * k * (80 + 273))

H_in_F  = SRIM("Computations/Profile/stopping/H_in_F.stop")
H_in_H  = SRIM("Computations/Profile/stopping/H_in_H.stop")
H_in_Fe = SRIM("Computations/Profile/stopping/H_in_Fe.stop")
H_in_Ta = SRIM("Computations/Profile/stopping/H_in_Ta.stop")
H_in_Ca = SRIM("Computations/Profile/stopping/H_in_Ca.stop")
H_in_Li = SRIM("Computations/Profile/stopping/H_in_Li.stop")


def effective_stopping_CaF2(energy, n_inactive=1, n_active=2):
    return H_in_F.eval(energy) + (n_inactive / n_active) * H_in_Ca.eval(energy)

def effective_stopping_Fe(energy, n_inactive=1, n_active=3):
    return H_in_F.eval(energy) + (n_inactive / n_active) * H_in_Fe.eval(energy)

def effective_stopping_Ta(energy, n_inactive=1, n_active=5):
    return H_in_F.eval(energy) + (n_inactive / n_active) * H_in_Ta.eval(energy)

def effective_stopping_Li(energy, n_inactive=1, n_active=1):
    return H_in_F.eval(energy) + (n_inactive / n_active) * H_in_Li.eval(energy)


popt = [0.75324712]

def straggling(x):
    return popt[0] * np.sqrt(x)

# =============================================================================
#  FUNZIONI DI PROFILO  (identiche all'originale)
# =============================================================================
def gaussian(x, x0, s):
    return np.exp(-(x - x0) ** 2 / (s * s * 2))


_skg_cache = {}   # cache locale a ogni processo — nessun problema con Pool

def skewed_gaussian(x, x0, s, alpha):
    if s <= 0:
        return np.zeros_like(x)
    arr = np.array(x)
    raw = skewnorm.pdf(arr, a=alpha, loc=x0, scale=s)
    key = (float(x0), float(s), float(alpha))
    if key not in _skg_cache:
        grid   = np.linspace(x0 - 10 * s, x0 + 10 * s, 2001)
        raw_g  = skewnorm.pdf(grid, a=alpha, loc=x0, scale=s)
        max_raw = np.max(raw_g)
        _skg_cache[key] = 1.0 / max_raw if max_raw > 0 else 1.0
    return raw * _skg_cache[key]


def profile(de, theta, target_type):
    """theta può essere lmfit.Parameters oppure un dict plain."""
    if target_type == 'implanted' and target != 'LiF':
        if de <= 0:
            return 0
        return skewed_gaussian(de, theta["mean"], theta["std"], theta["alpha"])
    elif (target_type == 'implanted' or target_type == 'evaporated') and (target == 'LiF' or target == 'EVA_AT#3'):
        edge = 0.05
        sq2  = np.sqrt(2)
        w    = theta["width"]
        s0 = 0.5 * (1 + erf(de          / (sq2 * edge)))
        s1 = 0.5 * (1 + erf((de - w)   / (sq2 * edge)))
        return s0 - s1
    elif target_type == 'fluorinated':
        edge = 0.05
        sq2  = np.sqrt(2)
        w1   = theta["width1"]
        w2   = theta["width1"] + theta["width2"]
        w3   = theta["width1"] + theta["width2"] + theta["width3"]
        s0 = 0.5 * (1 + erf(de          / (sq2 * edge)))
        s1 = 0.5 * (1 + erf((de - w1)   / (sq2 * edge)))
        s2 = 0.5 * (1 + erf((de - w2)   / (sq2 * edge)))
        s3 = 0.5 * (1 + erf((de - w3)   / (sq2 * edge)))
        return (s0 - s1) + theta["norm1"] * (s1 - s2) + theta["norm2"] * (s2 - s3)

# =============================================================================
#  CROSS-SECTION E YIELD
# =============================================================================
def cross_section(x0, theta, de):
    return np.interp(x0, energies * 1e3, extrap)


def reaction_yield(x0, theta, de, target_type, backing):
    cross = cross_section(x0, theta, de) * 1e-24
    nb    = theta["n_backing"]
    nf    = theta["n_f"]
    if backing == 'Ta' and target != 'LiF' and target != 'EVA_AT#3':
        stop = effective_stopping_Ta(x0, nb, nf) * 1e-15 * 1e-3
    elif backing == 'Ta' and target == 'LiF':
        stop = effective_stopping_Li(x0, nb, nf) * 1e-15 * 1e-3
    elif backing == 'Ca' and target == 'EVA_AT#3':
        stop = effective_stopping_CaF2(x0, nb, nf) * 1e-15 * 1e-3
    else:
        stop = effective_stopping_Fe(x0, nb, nf) * 1e-15 * 1e-3
    p = profile(de, theta, target_type)
    return cross / stop * p


def integrand(x, theta, x0, target_type, backing):
    de  = x0 - x
    s   = np.sqrt(theta["beam"] ** 2 + dopp ** 2)
    if de > 0:
        s = np.sqrt((theta["strag"] * straggling(de)) ** 2 + s ** 2)
    arr   = np.linspace(x - 3 * s, x + 3 * s, 100)
    gauss = gaussian(arr, x, s)
    step  = arr[1] - arr[0]
    norm  = np.sum(gauss) * step
    conv  = np.sum(gauss * reaction_yield(arr, theta, de, target_type, backing)) * step / norm
    return conv

# =============================================================================
#  HELPER: converti Parameters → dict (necessario per pickling nei worker)
# =============================================================================
def params_to_dict(params):
    """Restituisce un dict {name: value} da lmfit.Parameters."""
    if isinstance(params, dict):
        return params
    return {name: p.value for name, p in params.items()}

# =============================================================================
#  WORKER per un singolo punto di straggled_profile
# =============================================================================
def _integrand_at_x0(args):
    """
    Worker eseguito dal Pool.
    args = (x0, theta_dict, target_type, backing, nsteps)
    Restituisce lo scalare integrate.simpson per quel punto x0.
    """
    x0, theta_dict, target_type, backing, nsteps = args
    xmin   = x0 - 100
    xmax   = x0
    dx     = abs(xmax - xmin) / nsteps
    x_pts  = np.linspace(xmin, xmax, nsteps)
    values = [integrand(xi, theta_dict, x0, target_type, backing) for xi in x_pts]
    return integrate.simpson(values, dx=dx)

# =============================================================================
#  straggled_profile  —  versione parallela e versione seriale
# =============================================================================
def straggled_profile_parallel(x, theta, target_type, backing,
                                nsteps=300, n_workers=N_WORKERS):
    """Calcola straggled_profile parallelizzando su i punti di x."""
    theta_dict = params_to_dict(theta)
    job_args   = [(x0, theta_dict, target_type, backing, nsteps) for x0 in x]

    with Pool(processes=n_workers) as pool:
        results = pool.map(_integrand_at_x0, job_args)

    return np.array(results)


def straggled_profile_serial(x, theta, target_type, backing, nsteps=300):
    """Versione seriale originale (usata dentro i worker di scan-level)."""
    theta_dict = params_to_dict(theta)
    y = np.zeros(len(x))
    for idx, x0 in enumerate(x):
        xmin   = x0 - 100
        xmax   = x0
        dx     = abs(xmax - xmin) / nsteps
        x_pts  = np.linspace(xmin, xmax, nsteps)
        values = [integrand(xi, theta_dict, x0, target_type, backing) for xi in x_pts]
        y[idx] = integrate.simpson(values, dx=dx)
    return y


def straggled_profile(x, theta, target_type, backing):
    """Dispatcher: usa la versione parallela o seriale in base al flag globale
    e a se siamo già dentro un worker (evita Pool annidati)."""
    in_worker = mp.current_process().name != "MainProcess"
    if PROFILE_LEVEL_PARALLEL and not in_worker:
        return straggled_profile_parallel(x, theta, target_type, backing)
    else:
        return straggled_profile_serial(x, theta, target_type, backing)

# =============================================================================
#  MODEL e CHI2
# =============================================================================
def model(x, theta, target_type, backing):
    return straggled_profile(x, theta, target_type, backing)


_silent = False

def chi2(params, x, y, y_err, eff, target_type, backing):
    mod = model(x, params, target_type, backing) / q_e / 1e6 * eff
    res = (y - mod) / y_err
    if not _silent:
        print("Chi2: {:10.4f}".format(np.sum(res ** 2)), end="\r")
    return res

# =============================================================================
#  HELPER: salva parametri LM (e opzionalmente MCMC) su file .txt
# =============================================================================
def save_params_txt(filepath, free_params, out_lm, mcmc_flat=None):
    """
    Saves LM and (optionally) MCMC fit results to a tab-separated txt file.

    Layout
    ------
    Row 1 (header) : param0  param0_min  param0_max  param1  param1_min  ...
    Row 2 (LM)     : value   val-stderr  val+stderr  ...   (symmetric bounds)
    Row 3 (MCMC)   : q50     q16         q84         ...   (asymmetric bounds)
                     → row 3 only written when mcmc_flat is not None
    """
    # ── header ───────────────────────────────────────────────────────────────
    header_cols = []
    for name in free_params:
        header_cols += [name, f"{name}_min", f"{name}_max"]
    header_line = "\t".join(header_cols)

    # ── LM row: symmetric errors ──────────────────────────────────────────────
    lm_cols = []
    for name in free_params:
        p      = out_lm.params[name]
        val    = p.value  if p.value  is not None else float('nan')
        stderr = p.stderr if p.stderr is not None else float('nan')
        lm_cols += [
            f"{val:.6g}",
            f"{val - stderr:.6g}",
            f"{val + stderr:.6g}",
        ]
    lm_line = "\t".join(lm_cols)

    lines = [header_line, lm_line]

    # ── MCMC row: asymmetric errors from percentiles ──────────────────────────
    if mcmc_flat is not None:
        mcmc_cols = []
        for i, name in enumerate(free_params):
            q16, q50, q84 = np.percentile(mcmc_flat[:, i], [16, 50, 84])
            mcmc_cols += [
                f"{q50:.6g}",
                f"{q16:.6g}",
                f"{q84:.6g}",
            ]
        lines.append("\t".join(mcmc_cols))

    with open(filepath, "w") as f:
        f.write("\n".join(lines) + "\n")

# =============================================================================
#  FUNZIONE CHE PROCESSA UN SINGOLO (target, scan)  ← usata da scan-level Pool
# =============================================================================
def process_scan(job):
    """
    Eseguita (eventualmente) da un worker Pool.
    job è un dict con tutte le info necessarie.
    Restituisce un dict con i risultati del fit (o None se fallisce).
    """
    global _silent

    target       = job["target"]
    scan_label   = job["scan_label"]
    x            = job["x"]
    y            = job["y"]
    y_err        = job["y_err"]
    target_type  = job["target_type"]
    backing_type = job["backing_type"]
    results_dir  = job["results_dir"]

    # ── ricostruisci Parameters ──────────────────────────────────────────────
    if target_type == "implanted" and target != "LiF":
        params = Parameters()
        params.add("beam",      value=0.12, vary=False)
        params.add("strag",     value=1,    vary=False)
        params.add("n_backing", value=1.0,  vary=True,  min=0.0, max=30.0)
        params.add("n_f",       value=1.0,  vary=False)
        params.add("mean",      value=2.0,  vary=True,  min=0.0, max=10.0)
        params.add("std",       value=7.0,  vary=True,  min=0.0, max=15.0)
        params.add("alpha",     value=0.0,  vary=False, min=0.0, max=7.5)
    elif target_type == "implanted" and target == "LiF":
        params = Parameters()
        params.add("beam",      value=0.12, vary=False)
        params.add("strag",     value=1,    vary=False, min=0.9,  max=1.1)
        params.add("n_backing", value=2.5,  vary=True,  min=0.0,  max=30.0)
        params.add("n_f",       value=1.0,  vary=False)
        params.add("width",     value=8.0,  vary=True,  min=1.0,  max=80.0)
    elif target_type == "evaporated" and target == "EVA_AT#3":
        params = Parameters()
        params.add("beam",      value=0.12, vary=False)
        params.add("strag",     value=1,    vary=False, min=0.9,  max=1.1)
        params.add("n_backing", value=2.5,  vary=True,  min=0.0,  max=30.0)
        params.add("n_f",       value=1.0,  vary=False)
        params.add("width",     value=8.0,  vary=True,  min=1.0,  max=80.0)
    else:
        params = Parameters()
        params.add("beam",      value=0.12, vary=False)
        params.add("strag",     value=1,    vary=False, min=0.9,  max=1.1)
        params.add("n_backing", value=2.5,  vary=True,  min=0.0,  max=30.0)
        params.add("n_f",       value=1.0,  vary=False)
        params.add("width1",    value=8.0,  vary=True,  min=1.0,  max=80.0)
        params.add("width2",    value=10.0, vary=True,  min=1.0,  max=80.0)
        params.add("width3",    value=20.0, vary=True,  min=1.0,  max=80.0)
        params.add("norm1",     value=0.3,  vary=True,  min=0.0,  max=1.0)
        params.add("norm2",     value=0.1,  vary=True,  min=0.0,  max=1.0)

    eff = eff_far

    # ── sostituisci errori zero ──────────────────────────────────────────────
    y_err_orig = y_err.copy()
    if np.any(y_err == 0):
        nz  = y_err[y_err > 0]
        rep = np.median(nz) if nz.size else 1.0
        y_err_orig[y_err_orig == 0] = rep

    bias_mask = (x > 340)  # Bias calcolato sui punti della coda
                                      # e converto 340keV (lab) in (cm) perché i punti
                                      # sperimentali sono convertiti appena vengono letti

    # ── safe filename stem shared by plot and params files ───────────────────
    safe_stem = "".join(
        c if (c.isalnum() or c in ('_', '-')) else '_'
        for c in f"{target}_{scan_label}_340"
    )
    params_path = os.path.join(results_dir, f"{safe_stem}_params.txt")

    try:
        f_bias   = 0.35
        tol      = 0.02
        max_iter = 10
        for it in range(max_iter):
            bias      = np.mean(y[bias_mask]) * f_bias
            y_err_fit = y_err_orig + bias
            out = minimize(chi2, params,
                           args=(x, y, y_err_fit, eff, target_type, backing_type),
                           max_nfev=5000)
            redchi = out.redchi if (out.nfree > 0 and np.isfinite(out.redchi)) else float('nan')
            if np.isnan(redchi) or abs(redchi - 1.0) < tol:
                break
            f_bias *= min(np.sqrt(redchi), 3.0)

        print(f"\nFit results for {target} {scan_label}:")
        for name, p in out.params.items():
            stderr = p.stderr if p.stderr is not None else float('nan')
            print(f"  {name}: {p.value:.6g} +/- {stderr:.6g} (vary={p.vary})")
        print(f"  → redchi2 = {redchi:.4f}, f_bias = {f_bias:.4f} (converged in {it+1} iterations)")

        # ── collect free parameter names (shared by LM save and MCMC) ────────
        free_params = [n for n, p in out.params.items() if p.vary]

        # ── save LM-only result immediately ──────────────────────────────────
        # (will be overwritten with LM+MCMC below if MCMC succeeds)
        save_params_txt(params_path, free_params, out, mcmc_flat=None)
        print(f"LM params saved:      {params_path}")

        # mcmc_flat and mcmc_best_params are set inside the MCMC block when it
        # succeeds; the plot block reads them to optionally draw the MCMC curve.
        mcmc_flat        = None   # np.ndarray (nsamples, nfree) or None
        mcmc_best_params = None   # lmfit.Parameters built from MCMC medians or None

        # ── MCMC (sempre seriale) ─────────────────────────────────────────────
        if RUN_MCMC:
            _silent = True
            nwalkers = max(MCMC_NWALKERS, 2 * len(free_params) + 2)
            try:
                out_mcmc = minimize(
                    chi2, out.params, method='emcee',
                    args=(x, y, y_err_fit, eff, target_type, backing_type),
                    nan_policy='omit',
                    burn=MCMC_BURN, steps=MCMC_STEPS, thin=MCMC_THIN,
                    nwalkers=nwalkers, is_weighted=True, progress=True,
                )
                _silent = False
                flat      = out_mcmc.flatchain[free_params].to_numpy()
                mcmc_flat = flat   # expose to plot block

                # ── build a Parameters object from MCMC medians for plotting ──
                mcmc_best_params = out.params.copy()
                for i, name in enumerate(free_params):
                    mcmc_best_params[name].value = float(np.median(flat[:, i]))

                # ── save MCMC samples as CSV ──────────────────────────────────
                mcmc_df   = pd.DataFrame(flat, columns=free_params)
                mcmc_name = "".join(
                    c if (c.isalnum() or c in ('_', '-')) else '_'
                    for c in f"{target}_{scan_label}_{ENERGY_TAG}_mcmc_samples"
                )
                mcmc_df.to_csv(os.path.join(results_dir, f"{mcmc_name}.csv"), index=False)
                print(f"MCMC samples saved:   {mcmc_name}.csv")

                # ── print credible intervals ──────────────────────────────────
                print(f"\nMCMC credible intervals ({target} {scan_label}):")
                for i, name in enumerate(free_params):
                    q16, q50, q84 = np.percentile(flat[:, i], [16, 50, 84])
                    print(f"  {name}: {q50:.4g}  +{q84-q50:.4g} / -{q50-q16:.4g}")

                # ── corner plot ───────────────────────────────────────────────
                truths = [out.params[n].value for n in free_params]
                fig_c  = corner.corner(flat, labels=free_params,
                                       quantiles=[0.16, 0.5, 0.84],
                                       show_titles=True, title_fmt='.3g',
                                       truth_color='royalblue', truths=truths)
                cname = "".join(
                    c if (c.isalnum() or c in ('_', '-')) else '_'
                    for c in f"{target}_{scan_label}_{ENERGY_TAG}_corner"
                )
                fig_c.savefig(os.path.join(results_dir, f"{cname}.png"),
                              dpi=150, bbox_inches='tight')
                plt.close(fig_c)
                print(f"Corner plot saved:    {cname}.png")

                # ── overwrite params.txt with LM + MCMC rows ─────────────────
                save_params_txt(params_path, free_params, out, mcmc_flat=mcmc_flat)
                print(f"LM+MCMC params saved: {params_path}")

            except Exception as e_mcmc:
                _silent = False
                print(f"MCMC failed for {target} {scan_label}: {e_mcmc}")
                # params_path already holds the LM-only result — nothing to do

        # ── plot ──────────────────────────────────────────────────────────────
        try:
            fig, ax = plt.subplots(figsize=(8, 6))

            # all data points (black circles with error bars)
            ax.errorbar(x, y, yerr=y_err_fit, fmt='o', label='Data',
                        color='black', capsize=4, zorder=2)

            # bias points highlighted as orange diamonds on top
            if np.any(bias_mask):
                ax.scatter(x[bias_mask], y[bias_mask],
                           marker='o', color='none', s=150, edgecolor='forestgreen',
                           label='Bias points', zorder=3)

            # model grid
            grid = np.linspace(np.min(x) * 0.98, np.max(x) * 1.02, 300)

            # LM fit — solid deeppink
            y_mod_lm = model(grid, out.params, target_type, backing_type) / q_e / 1e6 * eff
            ax.plot(grid, y_mod_lm, ls='solid', lw=2, color='deeppink',
                    label='Fit', zorder=4)

            ax.set_title(f"{target} {scan_label}")
            ax.set_xlabel(r'$E_p^{\mathrm{cm}}$ [keV]')
            ax.set_ylabel(r'Yield [counts/$\mu$C]')
            ax.set_ylim(0, None)
            ax.grid()
            ax.legend(loc='upper left', fontsize=11)

            ax2 = plt.twinx()
            grid_profile = np.linspace(-10, 60, 1000)
            p_vals = [profile(de, out.params, target_type) for de in grid_profile]
            p_vals = np.array(p_vals)
            grid_profile += 323
            ax2.plot(grid_profile, p_vals, ls='solid', lw=2,
                     label='Profile', color='royalblue')
            ax2.fill_between(grid_profile, p_vals, 0,
                             facecolor='None', edgecolor='royalblue',
                             hatch='///////', linewidth=0)
            ax2.set_yticks([])
            ax2.set_ylabel('')
            ax2.set_xlim(310, 360)
            ax2.set_ylim(0, 2)

            outpath = os.path.join(results_dir, f"{safe_stem}.png")
            fig.savefig(outpath, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved plot:           {outpath}")
        except Exception as e:
            print(f"Could not generate plot for {target} {scan_label}: {e}")

        # ── raccogli risultati (solo implanted) ───────────────────────────────
        if target_type == 'implanted':
            def val_and_err(pname):
                if pname in out.params:
                    p = out.params[pname]
                    v = p.value  if p.value  is not None else float('nan')
                    e = p.stderr if p.stderr is not None else float('nan')
                else:
                    v, e = float('nan'), float('nan')
                return float(v), float(e)

            n_backing, n_backing_err = val_and_err('n_backing')
            mean,      mean_err      = val_and_err('mean')
            std,       std_err       = val_and_err('std')
            alpha,     alpha_err     = val_and_err('alpha')

            return {
                'target':          target,
                'scan':            scan_label,
                'energy_keV':      340,
                'n_backing':       n_backing,
                'n_backing_error': n_backing_err,
                'mean':            mean,
                'mean_error':      mean_err,
                'std':             std,
                'std_error':       std_err,
                'alpha':           alpha,
                'alpha_error':     alpha_err,
                'chi2':            out.chisqr,
            }

    except Exception as e:
        print(f"Fit failed for {target} {scan_label}: {e}")

    return None

# =============================================================================
#  MAIN
# =============================================================================
if __name__ == "__main__":
    # ── lista target ──────────────────────────────────────────────────────────
    targets      = ['EVA_AT#3', 'IMP_LTA#1', 'IMP_LTA#2', 'IMP_LTA#3', 'IMP_LTA#4',
                    'IMP_LTA-Low#1', 'IMP_LFE#1', 'IMP_LFE#2', 'IMP_LFE#3', 'IMP_LFE-Low#1']
    backings     = ['Ca', 'Ta', 'Ta', 'Ta', 'Ta', 'Ta', 'Fe', 'Fe', 'Fe', 'Fe']
    target_types = ['evaporated', 'implanted', 'implanted', 'implanted', 'implanted',
                    'implanted', 'implanted', 'implanted', 'implanted', 'implanted']

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)

    # ── costruisci lista job ───────────────────────────────────────────────────
    jobs = []
    for target_idx, target in enumerate(targets):
        target_type  = target_types[target_idx]
        backing_type = backings[target_idx]

        csv_path = f"Yield_scans/Results/Yield_{target}.csv"
        df = pd.read_csv(csv_path)
        if "Scan" not in df.columns:
            print(f"No 'Scan' column in {csv_path}, skipping")
            continue

        for scan_label in df["Scan"].unique():
            df_scan = df[df["Scan"] == scan_label]
            x = df_scan["Energy"].to_numpy()
            y = df_scan["Yield"].to_numpy()
            y_err = (df_scan["Yield Error"].to_numpy()
                     if "Yield Error" in df_scan.columns
                     else np.ones_like(y))

            mask = (~np.isnan(x)) & (x > 300)
            if not np.any(mask):
                print(f"{target} {scan_label}: no E>300 keV points — skipping")
                continue

            x     = x[mask]
            y     = y[mask]
            y_err = y_err[mask]

            # Converti in CM frame
            x = x * 19 / 20.007

            # Ordina per energia crescente
            idx_sort = np.argsort(x)
            x, y, y_err = x[idx_sort], y[idx_sort], y_err[idx_sort]

            jobs.append({
                "target":       target,
                "scan_label":   scan_label,
                "x":            x,
                "y":            y,
                "y_err":        y_err,
                "target_type":  target_type,
                "backing_type": backing_type,
                "results_dir":  results_dir,
            })

    # ── esegui i job ──────────────────────────────────────────────────────────
    if SCAN_LEVEL_PARALLEL and not PROFILE_LEVEL_PARALLEL:
        # Parallelizza il loop su (target, scan)
        # PROFILE_LEVEL_PARALLEL deve essere False per evitare Pool annidati
        print(f"Modalità: scan-level parallel ({len(jobs)} job, {N_WORKERS or mp.cpu_count()} worker)")
        with Pool(processes=N_WORKERS) as pool:
            all_results = pool.map(process_scan, jobs)
    else:
        # Seriale sul loop scan; la parallelizzazione avviene dentro
        # straggled_profile se PROFILE_LEVEL_PARALLEL=True
        mode = "profile-level parallel" if PROFILE_LEVEL_PARALLEL else "fully serial"
        print(f"Modalità: {mode} ({len(jobs)} job)")
        all_results = [process_scan(job) for job in jobs]

    # ── salva risultati aggregati ──────────────────────────────────────────────
    fit_results = [r for r in all_results if r is not None]
    if fit_results:
        results_df = pd.DataFrame(fit_results)
        out_csv    = os.path.join(results_dir, "fit_params_340.csv")
        results_df.to_csv(out_csv, index=False)
        print(f"Fit results saved to {out_csv}")
    else:
        print("No fit results were collected; nothing to save.")
MCMC_NWALKERS = 8
MCMC_BURN     = 100
MCMC_STEPS    = 300
MCMC_THIN     = 5
ENERGY_TAG    = "340"

# =============================================================================
#  COSTANTI
# =============================================================================
k           = 8.617e-5
eff_far     = 0.0044
eff_close   = 0.689
q_e         = 1.602176634e-19

# =============================================================================
#  DATI E STOPPING
# =============================================================================
from pyazr import azure2
from SRIM import SRIM

data     = np.loadtxt("Computations/Profile/utils/all_imp.extrap", usecols=(0, 3))
energies = data[:, 0]
extrap   = data[:, 1]

dopp = np.sqrt(2 * 1.007 / 19 * 0.250 * k * (80 + 273))

H_in_F  = SRIM("Computations/Profile/stopping/H_in_F.stop")
H_in_H  = SRIM("Computations/Profile/stopping/H_in_H.stop")
H_in_Fe = SRIM("Computations/Profile/stopping/H_in_Fe.stop")
H_in_Ta = SRIM("Computations/Profile/stopping/H_in_Ta.stop")
H_in_Ca = SRIM("Computations/Profile/stopping/H_in_Ca.stop")
H_in_Li = SRIM("Computations/Profile/stopping/H_in_Li.stop")


def effective_stopping_CaF2(energy, n_inactive=1, n_active=2):
    return H_in_F.eval(energy) + (n_inactive / n_active) * H_in_Ca.eval(energy)

def effective_stopping_Fe(energy, n_inactive=1, n_active=3):
    return H_in_F.eval(energy) + (n_inactive / n_active) * H_in_Fe.eval(energy)

def effective_stopping_Ta(energy, n_inactive=1, n_active=5):
    return H_in_F.eval(energy) + (n_inactive / n_active) * H_in_Ta.eval(energy)

def effective_stopping_Li(energy, n_inactive=1, n_active=1):
    return H_in_F.eval(energy) + (n_inactive / n_active) * H_in_Li.eval(energy)


popt = [0.75324712]

def straggling(x):
    return popt[0] * np.sqrt(x)

# =============================================================================
#  FUNZIONI DI PROFILO  (identiche all'originale)
# =============================================================================
def gaussian(x, x0, s):
    return np.exp(-(x - x0) ** 2 / (s * s * 2))


_skg_cache = {}   # cache locale a ogni processo — nessun problema con Pool

def skewed_gaussian(x, x0, s, alpha):
    if s <= 0:
        return np.zeros_like(x)
    arr = np.array(x)
    raw = skewnorm.pdf(arr, a=alpha, loc=x0, scale=s)
    key = (float(x0), float(s), float(alpha))
    if key not in _skg_cache:
        grid   = np.linspace(x0 - 10 * s, x0 + 10 * s, 2001)
        raw_g  = skewnorm.pdf(grid, a=alpha, loc=x0, scale=s)
        max_raw = np.max(raw_g)
        _skg_cache[key] = 1.0 / max_raw if max_raw > 0 else 1.0
    return raw * _skg_cache[key]


def profile(de, theta, target_type):
    """theta può essere lmfit.Parameters oppure un dict plain."""
    if target_type == 'implanted' and target != 'LiF':
        if de <= 0:
            return 0
        return skewed_gaussian(de, theta["mean"], theta["std"], theta["alpha"])
    elif (target_type == 'implanted' or target_type == 'evaporated') and (target == 'LiF' or target == 'EVA_AT#3'):
        edge = 0.05
        sq2  = np.sqrt(2)
        w    = theta["width"]
        s0 = 0.5 * (1 + erf(de          / (sq2 * edge)))
        s1 = 0.5 * (1 + erf((de - w)   / (sq2 * edge)))
        return s0 - s1
    elif target_type == 'fluorinated':
        edge = 0.05
        sq2  = np.sqrt(2)
        w1   = theta["width1"]
        w2   = theta["width1"] + theta["width2"]
        w3   = theta["width1"] + theta["width2"] + theta["width3"]
        s0 = 0.5 * (1 + erf(de          / (sq2 * edge)))
        s1 = 0.5 * (1 + erf((de - w1)   / (sq2 * edge)))
        s2 = 0.5 * (1 + erf((de - w2)   / (sq2 * edge)))
        s3 = 0.5 * (1 + erf((de - w3)   / (sq2 * edge)))
        return (s0 - s1) + theta["norm1"] * (s1 - s2) + theta["norm2"] * (s2 - s3)

# =============================================================================
#  CROSS-SECTION E YIELD
# =============================================================================
def cross_section(x0, theta, de):
    return np.interp(x0, energies * 1e3, extrap)


def reaction_yield(x0, theta, de, target_type, backing):
    cross = cross_section(x0, theta, de) * 1e-24
    nb    = theta["n_backing"]
    nf    = theta["n_f"]
    if backing == 'Ta' and target != 'LiF' and target != 'EVA_AT#3':
        stop = effective_stopping_Ta(x0, nb, nf) * 1e-15 * 1e-3
    elif backing == 'Ta' and target == 'LiF':
        stop = effective_stopping_Li(x0, nb, nf) * 1e-15 * 1e-3
    elif backing == 'Ca' and target == 'EVA_AT#3':
        stop = effective_stopping_CaF2(x0, nb, nf) * 1e-15 * 1e-3
    else:
        stop = effective_stopping_Fe(x0, nb, nf) * 1e-15 * 1e-3
    p = profile(de, theta, target_type)
    return cross / stop * p


def integrand(x, theta, x0, target_type, backing):
    de  = x0 - x
    s   = np.sqrt(theta["beam"] ** 2 + dopp ** 2)
    if de > 0:
        s = np.sqrt((theta["strag"] * straggling(de)) ** 2 + s ** 2)
    arr   = np.linspace(x - 3 * s, x + 3 * s, 100)
    gauss = gaussian(arr, x, s)
    step  = arr[1] - arr[0]
    norm  = np.sum(gauss) * step
    conv  = np.sum(gauss * reaction_yield(arr, theta, de, target_type, backing)) * step / norm
    return conv

# =============================================================================
#  HELPER: converti Parameters → dict (necessario per pickling nei worker)
# =============================================================================
def params_to_dict(params):
    """Restituisce un dict {name: value} da lmfit.Parameters."""
    if isinstance(params, dict):
        return params
    return {name: p.value for name, p in params.items()}

# =============================================================================
#  WORKER per un singolo punto di straggled_profile
# =============================================================================
def _integrand_at_x0(args):
    """
    Worker eseguito dal Pool.
    args = (x0, theta_dict, target_type, backing, nsteps)
    Restituisce lo scalare integrate.simpson per quel punto x0.
    """
    x0, theta_dict, target_type, backing, nsteps = args
    xmin   = x0 - 100
    xmax   = x0
    dx     = abs(xmax - xmin) / nsteps
    x_pts  = np.linspace(xmin, xmax, nsteps)
    values = [integrand(xi, theta_dict, x0, target_type, backing) for xi in x_pts]
    return integrate.simpson(values, dx=dx)

# =============================================================================
#  straggled_profile  —  versione parallela e versione seriale
# =============================================================================
def straggled_profile_parallel(x, theta, target_type, backing,
                                nsteps=300, n_workers=N_WORKERS):
    """Calcola straggled_profile parallelizzando su i punti di x."""
    theta_dict = params_to_dict(theta)
    job_args   = [(x0, theta_dict, target_type, backing, nsteps) for x0 in x]

    with Pool(processes=n_workers) as pool:
        results = pool.map(_integrand_at_x0, job_args)

    return np.array(results)


def straggled_profile_serial(x, theta, target_type, backing, nsteps=300):
    """Versione seriale originale (usata dentro i worker di scan-level)."""
    theta_dict = params_to_dict(theta)
    y = np.zeros(len(x))
    for idx, x0 in enumerate(x):
        xmin   = x0 - 100
        xmax   = x0
        dx     = abs(xmax - xmin) / nsteps
        x_pts  = np.linspace(xmin, xmax, nsteps)
        values = [integrand(xi, theta_dict, x0, target_type, backing) for xi in x_pts]
        y[idx] = integrate.simpson(values, dx=dx)
    return y


def straggled_profile(x, theta, target_type, backing):
    """Dispatcher: usa la versione parallela o seriale in base al flag globale
    e a se siamo già dentro un worker (evita Pool annidati)."""
    in_worker = mp.current_process().name != "MainProcess"
    if PROFILE_LEVEL_PARALLEL and not in_worker:
        return straggled_profile_parallel(x, theta, target_type, backing)
    else:
        return straggled_profile_serial(x, theta, target_type, backing)

# =============================================================================
#  MODEL e CHI2
# =============================================================================
def model(x, theta, target_type, backing):
    return straggled_profile(x, theta, target_type, backing)


_silent = False

def chi2(params, x, y, y_err, eff, target_type, backing):
    mod = model(x, params, target_type, backing) / q_e / 1e6 * eff
    res = (y - mod) / y_err
    if not _silent:
        print("Chi2: {:10.4f}".format(np.sum(res ** 2)), end="\r")
    return res

# =============================================================================
#  HELPER: salva parametri LM (e opzionalmente MCMC) su file .txt
# =============================================================================
def save_params_txt(filepath, free_params, out_lm, mcmc_flat=None):
    """
    Saves LM and (optionally) MCMC fit results to a tab-separated txt file.

    Layout
    ------
    Row 1 (header) : param0  param0_min  param0_max  param1  param1_min  ...
    Row 2 (LM)     : value   val-stderr  val+stderr  ...   (symmetric bounds)
    Row 3 (MCMC)   : q50     q16         q84         ...   (asymmetric bounds)
                     → row 3 only written when mcmc_flat is not None
    """
    # ── header ───────────────────────────────────────────────────────────────
    header_cols = []
    for name in free_params:
        header_cols += [name, f"{name}_min", f"{name}_max"]
    header_line = "\t".join(header_cols)

    # ── LM row: symmetric errors ──────────────────────────────────────────────
    lm_cols = []
    for name in free_params:
        p      = out_lm.params[name]
        val    = p.value  if p.value  is not None else float('nan')
        stderr = p.stderr if p.stderr is not None else float('nan')
        lm_cols += [
            f"{val:.6f}",
            f"{val - stderr:.6f}",
            f"{val + stderr:.6f}",
        ]
    lm_line = "\t".join(lm_cols)

    lines = [header_line, lm_line]

    # ── MCMC row: asymmetric errors from percentiles ──────────────────────────
    if mcmc_flat is not None:
        mcmc_cols = []
        for i, name in enumerate(free_params):
            q16, q50, q84 = np.percentile(mcmc_flat[:, i], [16, 50, 84])
            mcmc_cols += [
                f"{q50:.6f}",
                f"{q16:.6f}",
                f"{q84:.6f}",
            ]
        lines.append("\t".join(mcmc_cols))

    with open(filepath, "w") as f:
        f.write("\n".join(lines) + "\n")

# =============================================================================
#  FUNZIONE CHE PROCESSA UN SINGOLO (target, scan)  ← usata da scan-level Pool
# =============================================================================
def process_scan(job):
    """
    Eseguita (eventualmente) da un worker Pool.
    job è un dict con tutte le info necessarie.
    Restituisce un dict con i risultati del fit (o None se fallisce).
    """
    global _silent

    target       = job["target"]
    scan_label   = job["scan_label"]
    x            = job["x"]
    y            = job["y"]
    y_err        = job["y_err"]
    target_type  = job["target_type"]
    backing_type = job["backing_type"]
    results_dir  = job["results_dir"]

    # ── ricostruisci Parameters ──────────────────────────────────────────────
    if target_type == "implanted" and target != "LiF":
        params = Parameters()
        params.add("beam",      value=0.12, vary=False)
        params.add("strag",     value=1,    vary=False)
        params.add("n_backing", value=1.0,  vary=True,  min=0.0, max=30.0)
        params.add("n_f",       value=1.0,  vary=False)
        params.add("mean",      value=2.0,  vary=True,  min=0.0, max=10.0)
        params.add("std",       value=7.0,  vary=True,  min=0.0, max=15.0)
        params.add("alpha",     value=0.0,  vary=False, min=0.0, max=7.5)
    elif target_type == "implanted" and target == "LiF":
        params = Parameters()
        params.add("beam",      value=0.12, vary=False)
        params.add("strag",     value=1,    vary=False, min=0.9,  max=1.1)
        params.add("n_backing", value=2.5,  vary=True,  min=0.0,  max=30.0)
        params.add("n_f",       value=1.0,  vary=False)
        params.add("width",     value=8.0,  vary=True,  min=1.0,  max=80.0)
    elif target_type == "evaporated" and target == "EVA_AT#3":
        params = Parameters()
        params.add("beam",      value=0.12, vary=False)
        params.add("strag",     value=1,    vary=False, min=0.9,  max=1.1)
        params.add("n_backing", value=2.5,  vary=True,  min=0.0,  max=30.0)
        params.add("n_f",       value=1.0,  vary=False)
        params.add("width",     value=8.0,  vary=True,  min=1.0,  max=80.0)
    else:
        params = Parameters()
        params.add("beam",      value=0.12, vary=False)
        params.add("strag",     value=1,    vary=False, min=0.9,  max=1.1)
        params.add("n_backing", value=2.5,  vary=True,  min=0.0,  max=30.0)
        params.add("n_f",       value=1.0,  vary=False)
        params.add("width1",    value=8.0,  vary=True,  min=1.0,  max=80.0)
        params.add("width2",    value=10.0, vary=True,  min=1.0,  max=80.0)
        params.add("width3",    value=20.0, vary=True,  min=1.0,  max=80.0)
        params.add("norm1",     value=0.3,  vary=True,  min=0.0,  max=1.0)
        params.add("norm2",     value=0.1,  vary=True,  min=0.0,  max=1.0)

    eff = eff_far

    # ── sostituisci errori zero ──────────────────────────────────────────────
    y_err_orig = y_err.copy()
    if np.any(y_err == 0):
        nz  = y_err[y_err > 0]
        rep = np.median(nz) if nz.size else 1.0
        y_err_orig[y_err_orig == 0] = rep

    bias_mask = (x > 340)  # Bias calcolato sui punti della coda

    # ── safe filename stem shared by plot and params files ───────────────────
    safe_stem = "".join(
        c if (c.isalnum() or c in ('_', '-')) else '_'
        for c in f"{target}_{scan_label}_340"
    )
    params_path = os.path.join(results_dir, f"{safe_stem}_params.txt")

    try:
        f_bias   = 0.35
        tol      = 0.02
        max_iter = 10
        for it in range(max_iter):
            bias      = np.mean(y[bias_mask]) * f_bias
            y_err_fit = y_err_orig + bias
            out = minimize(chi2, params,
                           args=(x, y, y_err_fit, eff, target_type, backing_type),
                           max_nfev=5000)
            redchi = out.redchi if (out.nfree > 0 and np.isfinite(out.redchi)) else float('nan')
            if np.isnan(redchi) or abs(redchi - 1.0) < tol:
                break
            f_bias *= min(np.sqrt(redchi), 3.0)

        print(f"\nFit results for {target} {scan_label}:")
        for name, p in out.params.items():
            stderr = p.stderr if p.stderr is not None else float('nan')
            print(f"  {name}: {p.value:.6g} +/- {stderr:.6g} (vary={p.vary})")
        print(f"  → redchi2 = {redchi:.4f}, f_bias = {f_bias:.4f} (converged in {it+1} iterations)")

        # ── collect free parameter names (shared by LM save and MCMC) ────────
        free_params = [n for n, p in out.params.items() if p.vary]

        # ── save LM-only result immediately ──────────────────────────────────
        # (will be overwritten with LM+MCMC below if MCMC succeeds)
        save_params_txt(params_path, free_params, out, mcmc_flat=None)
        print(f"LM params saved:      {params_path}")

        # mcmc_flat and mcmc_best_params are set inside the MCMC block when it
        # succeeds; the plot block reads them to optionally draw the MCMC curve.
        mcmc_flat        = None   # np.ndarray (nsamples, nfree) or None
        mcmc_best_params = None   # lmfit.Parameters built from MCMC medians or None

        # ── MCMC (sempre seriale) ─────────────────────────────────────────────
        if RUN_MCMC:
            _silent = True
            nwalkers = max(MCMC_NWALKERS, 2 * len(free_params) + 2)
            try:
                out_mcmc = minimize(
                    chi2, out.params, method='emcee',
                    args=(x, y, y_err_fit, eff, target_type, backing_type),
                    nan_policy='omit',
                    burn=MCMC_BURN, steps=MCMC_STEPS, thin=MCMC_THIN,
                    nwalkers=nwalkers, is_weighted=True, progress=True,
                )
                _silent = False
                flat      = out_mcmc.flatchain[free_params].to_numpy()
                mcmc_flat = flat   # expose to plot block

                # ── build a Parameters object from MCMC medians for plotting ──
                mcmc_best_params = out.params.copy()
                for i, name in enumerate(free_params):
                    mcmc_best_params[name].value = float(np.median(flat[:, i]))

                # ── save MCMC samples as CSV ──────────────────────────────────
                mcmc_df   = pd.DataFrame(flat, columns=free_params)
                mcmc_name = "".join(
                    c if (c.isalnum() or c in ('_', '-')) else '_'
                    for c in f"{target}_{scan_label}_{ENERGY_TAG}_mcmc_samples"
                )
                mcmc_df.to_csv(os.path.join(results_dir, f"{mcmc_name}.csv"), index=False)
                print(f"MCMC samples saved:   {mcmc_name}.csv")

                # ── print credible intervals ──────────────────────────────────
                print(f"\nMCMC credible intervals ({target} {scan_label}):")
                for i, name in enumerate(free_params):
                    q16, q50, q84 = np.percentile(flat[:, i], [16, 50, 84])
                    print(f"  {name}: {q50:.4g}  +{q84-q50:.4g} / -{q50-q16:.4g}")

                # ── corner plot ───────────────────────────────────────────────
                truths = [out.params[n].value for n in free_params]
                fig_c  = corner.corner(flat, labels=free_params,
                                       quantiles=[0.16, 0.5, 0.84],
                                       show_titles=True, title_fmt='.3g',
                                       truth_color='royalblue', truths=truths)
                cname = "".join(
                    c if (c.isalnum() or c in ('_', '-')) else '_'
                    for c in f"{target}_{scan_label}_{ENERGY_TAG}_corner"
                )
                fig_c.savefig(os.path.join(results_dir, f"{cname}.png"),
                              dpi=150, bbox_inches='tight')
                plt.close(fig_c)
                print(f"Corner plot saved:    {cname}.png")

                # ── overwrite params.txt with LM + MCMC rows ─────────────────
                save_params_txt(params_path, free_params, out, mcmc_flat=mcmc_flat)
                print(f"LM+MCMC params saved: {params_path}")

            except Exception as e_mcmc:
                _silent = False
                print(f"MCMC failed for {target} {scan_label}: {e_mcmc}")
                # params_path already holds the LM-only result — nothing to do

        # ── plot ──────────────────────────────────────────────────────────────
        try:
            fig, ax = plt.subplots(figsize=(8, 6))

            # all data points (black circles with error bars)
            ax.errorbar(x, y, yerr=y_err_fit, fmt='o', label='Data',
                        color='black', capsize=4, zorder=2)

            # bias points highlighted as orange diamonds on top
            if np.any(bias_mask):
                ax.scatter(x[bias_mask], y[bias_mask],
                           marker='o', color='none', s=120, edgecolor='forestgreen',
                           label='Bias points', zorder=3)

            # model grid
            grid = np.linspace(np.min(x) * 0.98, np.max(x) * 1.02, 300)

            # LM fit — solid deeppink
            y_mod_lm = model(grid, out.params, target_type, backing_type) / q_e / 1e6 * eff
            ax.plot(grid, y_mod_lm, ls='solid', lw=2, color='deeppink',
                    label='Fit', zorder=4)

            ax.set_title(f"{target} {scan_label}")
            ax.set_xlabel(r'$E_p^{\mathrm{cm}}$ [keV]')
            ax.set_ylabel(r'Yield [counts/$\mu$C]')
            ax.set_ylim(0, None)
            ax.grid()
            ax.legend(loc='upper left', fontsize=11)

            ax2 = plt.twinx()
            grid_profile = np.linspace(-10, 60, 1000)
            p_vals = [profile(de, out.params, target_type) for de in grid_profile]
            p_vals = np.array(p_vals)
            grid_profile += 323
            ax2.plot(grid_profile, p_vals, ls='solid', lw=2,
                     label='Profile', color='royalblue')
            ax2.fill_between(grid_profile, p_vals, 0,
                             facecolor='None', edgecolor='royalblue',
                             hatch='///////', linewidth=0)
            ax2.set_yticks([])
            ax2.set_ylabel('')
            ax2.set_xlim(310, 360)
            ax2.set_ylim(0, 2)

            outpath = os.path.join(results_dir, f"{safe_stem}.png")
            fig.savefig(outpath, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved plot:           {outpath}")
        except Exception as e:
            print(f"Could not generate plot for {target} {scan_label}: {e}")

        # ── raccogli risultati (solo implanted) ───────────────────────────────
        if target_type == 'implanted':
            def val_and_err(pname):
                if pname in out.params:
                    p = out.params[pname]
                    v = p.value  if p.value  is not None else float('nan')
                    e = p.stderr if p.stderr is not None else float('nan')
                else:
                    v, e = float('nan'), float('nan')
                return float(v), float(e)

            n_backing, n_backing_err = val_and_err('n_backing')
            mean,      mean_err      = val_and_err('mean')
            std,       std_err       = val_and_err('std')
            alpha,     alpha_err     = val_and_err('alpha')

            return {
                'target':          target,
                'scan':            scan_label,
                'energy_keV':      340,
                'n_backing':       n_backing,
                'n_backing_error': n_backing_err,
                'mean':            mean,
                'mean_error':      mean_err,
                'std':             std,
                'std_error':       std_err,
                'alpha':           alpha,
                'alpha_error':     alpha_err,
                'chi2':            out.chisqr,
            }

    except Exception as e:
        print(f"Fit failed for {target} {scan_label}: {e}")

    return None

# =============================================================================
#  MAIN
# =============================================================================
if __name__ == "__main__":
    # ── lista target ──────────────────────────────────────────────────────────
    targets      = ['EVA_AT#3', 'IMP_LTA#1', 'IMP_LTA#2', 'IMP_LTA#3', 'IMP_LTA#4',
                    'IMP_LTA-Low#1', 'IMP_LFE#1', 'IMP_LFE#2', 'IMP_LFE#3', 'IMP_LFE-Low#1']
    backings     = ['Ca', 'Ta', 'Ta', 'Ta', 'Ta', 'Ta', 'Fe', 'Fe', 'Fe', 'Fe']
    target_types = ['evaporated', 'implanted', 'implanted', 'implanted', 'implanted',
                    'implanted', 'implanted', 'implanted', 'implanted', 'implanted']

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)

    # ── costruisci lista job ───────────────────────────────────────────────────
    jobs = []
    for target_idx, target in enumerate(targets):
        target_type  = target_types[target_idx]
        backing_type = backings[target_idx]

        csv_path = f"Yield_scans/Results/Yield_{target}.csv"
        df = pd.read_csv(csv_path)
        if "Scan" not in df.columns:
            print(f"No 'Scan' column in {csv_path}, skipping")
            continue

        for scan_label in df["Scan"].unique():
            df_scan = df[df["Scan"] == scan_label]
            x = df_scan["Energy"].to_numpy()
            y = df_scan["Yield"].to_numpy()
            y_err = (df_scan["Yield Error"].to_numpy()
                     if "Yield Error" in df_scan.columns
                     else np.ones_like(y))

            mask = (~np.isnan(x)) & (x > 300)
            if not np.any(mask):
                print(f"{target} {scan_label}: no E>300 keV points — skipping")
                continue

            x     = x[mask]
            y     = y[mask]
            y_err = y_err[mask]

            # Converti in CM frame
            x = x * 19 / 20.007

            # Ordina per energia crescente
            idx_sort = np.argsort(x)
            x, y, y_err = x[idx_sort], y[idx_sort], y_err[idx_sort]

            jobs.append({
                "target":       target,
                "scan_label":   scan_label,
                "x":            x,
                "y":            y,
                "y_err":        y_err,
                "target_type":  target_type,
                "backing_type": backing_type,
                "results_dir":  results_dir,
            })

    # ── esegui i job ──────────────────────────────────────────────────────────
    if SCAN_LEVEL_PARALLEL and not PROFILE_LEVEL_PARALLEL:
        # Parallelizza il loop su (target, scan)
        # PROFILE_LEVEL_PARALLEL deve essere False per evitare Pool annidati
        print(f"Modalità: scan-level parallel ({len(jobs)} job, {N_WORKERS or mp.cpu_count()} worker)")
        with Pool(processes=N_WORKERS) as pool:
            all_results = pool.map(process_scan, jobs)
    else:
        # Seriale sul loop scan; la parallelizzazione avviene dentro
        # straggled_profile se PROFILE_LEVEL_PARALLEL=True
        mode = "profile-level parallel" if PROFILE_LEVEL_PARALLEL else "fully serial"
        print(f"Modalità: {mode} ({len(jobs)} job)")
        all_results = [process_scan(job) for job in jobs]

    # ── salva risultati aggregati ──────────────────────────────────────────────
    fit_results = [r for r in all_results if r is not None]
    if fit_results:
        results_df = pd.DataFrame(fit_results)
        out_csv    = os.path.join(results_dir, "fit_params_340.csv")
        results_df.to_csv(out_csv, index=False)
        print(f"Fit results saved to {out_csv}")
    else:
        print("No fit results were collected; nothing to save.")
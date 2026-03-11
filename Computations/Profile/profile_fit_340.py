# Import the required libraries

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
import os
import re

from scipy import integrate
from lmfit import Parameters, minimize

from scipy.special import gamma

from pyazr import azure2

from SRIM import SRIM
from scipy.special import erf
from scipy.stats import skewnorm

import emcee   # pip install emcee
import corner  # pip install corner

# Targets list
targets = ['IMP_LFE#1', 'IMP_LFE#2', 'IMP_LFE#3', 'IMP_LTA#1', 'IMP_LTA#2', 'SUDF#2', 'SUDF#3', 'SUDF#4']
backings = ['Fe', 'Fe', 'Fe', 'Ta', 'Ta', 'Ta', 'Ta', 'Ta']
target_types = ['implanted'] * 5 + ['fluorinated'] * 3

targets = ['IMP_LFE#1', 'IMP_LFE#2', 'IMP_LFE#3', 'IMP_LTA#1', 'IMP_LTA#2', 'IMP_LFE-Low#1', 'IMP_LTA-Low#1']
backings = ['Fe', 'Fe', 'Fe', 'Ta', 'Ta', 'Fe', 'Ta']
target_types = ['implanted'] * 7

# Test for imp_lfe low 1 only
targets = ['SUDF#4']
backings = ['Ta']
target_types = ['fluorinated']

# MCMC settings — set RUN_MCMC=True to run posterior sampling after the LM fit
# (slower, but gives corner plots and asymmetric credible intervals)
RUN_MCMC      = False  # toggle
MCMC_NWALKERS = 16     # must be >= 2 * n_free_params
MCMC_BURN     = 100    # burn-in steps to discard
MCMC_STEPS    = 300    # production steps
MCMC_THIN     = 5      # keep every N-th sample
ENERGY_TAG    = "340"  # used in output filenames

# Usefull constants
k = 8.617e-5            # Boltzmann constant in eV/K
eff_far = 0.0044        # Far geometry efficiency
eff_close = 0.689       # Close geometry efficiency
q_e = 1.602176634e-19   # Charge of the electron in C

# Define the cross-section
data = np.loadtxt( f"Computations/Profile/utils/all_imp.extrap", usecols=(0,3))
energies, extrap = data[:,0], data[:,1]

# Doppler effect
dopp = np.sqrt( 2 * 1.007 / 19 * 0.250 * k * ( 80 + 273 ) )

# Read the element stopping data
H_in_F = SRIM( "Computations/Profile/stopping/H_in_F.stop" )
H_in_H = SRIM( "Computations/Profile/stopping/H_in_H.stop" )
H_in_Fe = SRIM( "Computations/Profile/stopping/H_in_Fe.stop" )
H_in_Ta = SRIM( "Computations/Profile/stopping/H_in_Ta.stop" )
H_in_Ca = SRIM( "Computations/Profile/stopping/H_in_Ca.stop" )
H_in_Li = SRIM( "Computations/Profile/stopping/H_in_Li.stop" )

# Get the effective stopping power for CaF2
def effective_stopping_CaF2( energy, n_inactive=1, n_active=2 ):
    stoichiometry = n_inactive / n_active
    return H_in_F.eval( energy ) + stoichiometry * H_in_Ca.eval( energy )

# Get the effective stopping power for F implanted in Fe
def effective_stopping_Fe( energy, n_inactive=1, n_active=3 ):
    stoichiometry = n_inactive / n_active
    return H_in_F.eval( energy ) + stoichiometry * H_in_Fe.eval( energy )

# Get the effective stopping power for F implanted in Ta
def effective_stopping_Ta( energy, n_inactive=1, n_active=5 ):
    stoichiometry = n_inactive / n_active
    return H_in_F.eval( energy ) + stoichiometry * H_in_Ta.eval( energy )

# Get the effective stopping power for F implanted in Li
def effective_stopping_Li( energy, n_inactive=1, n_active=1 ):
    stoichiometry = n_inactive / n_active
    return H_in_F.eval( energy ) + stoichiometry * H_in_Li.eval( energy )

# Starggling
popt = [0.75324712]
def straggling( x ):
    return popt[0] * np.sqrt( x )

# Profile functions
def gaussian( x, x0, s ):
    return np.exp( -(x - x0)**2 / ( s*s*2 ) )

_skg_cache = {}

def skewed_gaussian(x, x0, s, alpha):
    if s <= 0:
        return np.zeros_like(x)
    # use scipy's skew-normal PDF (vectorized and optimized)
    arr = np.array(x)
    raw = skewnorm.pdf(arr, a=alpha, loc=x0, scale=s)

    # cache normalization factor per (x0,s,alpha)
    key = (float(x0), float(s), float(alpha))
    if key not in _skg_cache:
        grid = np.linspace(x0 - 10 * s, x0 + 10 * s, 2001)
        raw_g = skewnorm.pdf(grid, a=alpha, loc=x0, scale=s)
        max_raw = np.max(raw_g)
        _skg_cache[key] = 1.0 / max_raw if max_raw > 0 else 1.0

    return raw * _skg_cache[key]

def profile( de, theta, target_type ):
    if target_type == 'implanted':
        # Gaussian
        if de <= 0:
            return 0
        else:
            return skewed_gaussian( de, theta["mean"], theta["std"], theta["alpha"] )
            # return gaussian( de, theta["mean"]+theta["dead_layer"], theta["std"] )
    elif target_type == 'fluorinated':
        # --- 3-layer erf-smoothed profile (commented out) ---
        edge = 0.5  # keV smoothing scale at each boundary
        sq2 = np.sqrt(2)
        w1 = theta["width1"]
        w2 = theta["width1"] + theta["width2"]
        w3 = theta["width1"] + theta["width2"] + theta["width3"]
        s0 = 0.5 * (1 + erf( de          / (sq2 * edge)))  # rises at de=0
        s1 = 0.5 * (1 + erf((de - w1)    / (sq2 * edge)))  # rises at de=width1
        s2 = 0.5 * (1 + erf((de - w2)    / (sq2 * edge)))  # rises at de=width1+width2
        s3 = 0.5 * (1 + erf((de - w3)    / (sq2 * edge)))  # rises at de=width1+width2+width3
        return (s0 - s1) + theta["norm1"] * (s1 - s2) + theta["norm2"] * (s2 - s3)
    elif target_type == 'evaporated':
        # Pure single step
        if de <= 0 or de >= theta["width"]:
            return 0
        else:
            return 1
    elif target_type == 'implanted' and 'Low' in target:
        # Gaussian
        if de <= 0:
            return 0
        else:
            return gaussian( de, theta["mean"], theta["std"] )

# Cross-section function
def cross_section( x0, theta, de ):
    cross = np.interp( x0, energies * 1e3, extrap )
    return cross 

# Reaction yield function
def reaction_yield( x0, theta, de, target_type, backing ):
    cross = cross_section( x0, theta, de ) * 1e-24
    if backing == 'Ta':
        stop = effective_stopping_Ta( x0, theta["n_backing"], theta["n_f"] ) * 1e-15 * 1e-3
    elif backing == 'Fe':
        stop = effective_stopping_Fe( x0, theta["n_backing"], theta["n_f"] ) * 1e-15 * 1e-3
    elif backing == 'Li':
        stop = effective_stopping_Li( x0, theta["n_backing"], theta["n_f"] ) * 1e-15 * 1e-3
    elif backing == 'CaF2':
        stop = effective_stopping_CaF2( x0, theta["n_backing"], theta["n_f"] ) * 1e-15 * 1e-3
    p = profile( de, theta, target_type )
    return cross / stop * p

# Define the integrand for the convolution
def integrand( x, theta, x0, target_type, backing ):
    de = x0 - x

    # Get beam width
    s = np.sqrt( pow( theta["beam"], 2 ) + pow( dopp, 2 ) )
    s = np.sqrt( pow( theta["strag"] * straggling( de ), 2 ) + pow(s, 2) ) if de > 0 else s

    # Convolve the gaussian with the straggling
    array = np.linspace( x - 3 * s, x + 3 * s, 100 )
    gauss = gaussian( array, x, s )

    # Normalize the gaussian
    step = array[1] - array[0]
    norm  = np.sum( gauss ) * step

    # Convolve
    conv = np.sum( gauss * reaction_yield( array, theta, de, target_type, backing ) ) * step / norm

    return conv

# Define the straggled profile
def straggled_profile( x, theta, target_type, backing ):
    y = np.zeros( shape=len( x ) )
    for idx in range( len( x ) ):
        nsteps = 500
        xmin, xmax = x[idx] - 50, x[idx]
        dx = np.abs( xmax - xmin ) / nsteps
        y[idx] = integrate.simpson( [ integrand( x_i, theta, x[idx], target_type, backing ) for x_i in np.linspace( xmin, xmax, nsteps) ], dx=dx )
    return y

# Define the model
def model( x, theta, target_type, backing ):
    sign = straggled_profile( x, theta, target_type, backing )
    back = 0
    return sign + back

# Define the chi2 function
_silent = False   # set True during MCMC to suppress per-call progress print

def chi2( params, x, y, y_err, eff, target_type, backing ):
    mod   = model( x, params, target_type, backing ) / q_e / 1e6 * eff
    res   = ( y - mod ) / y_err
    if not _silent:
        print( "Chi2: {:10.4f}".format(np.sum(res**2)), end="\r" )
    return res

# Resolve results dir relative to this script so it works from any parent folder
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(results_dir, exist_ok=True)

# Loop over targets and scans, perform fits for scans with E<300 keV
# Collect fit results here
fit_results = []

for target_idx, target in enumerate(targets):
    target_type = target_types[target_idx]
    backing_type = backings[target_idx]

    # Initialize parameters per target type
    if target_type == "implanted":
        params = Parameters()
        params.add( "beam",  value=0.12, vary=False )
        params.add( "strag", value=1, vary=False, min=0.9, max=1.1  )
        params.add( "n_backing",  value=1.0, vary=True, min=0.0, max=7.0 )
        params.add( "n_f",   value=1.0, vary=False )
        params.add( "mean",  value=2.0, vary=True, min=0.0, max=10.0 )
        params.add( "std",   value=7.0, vary=True, min=0.0, max=10.0  )
        params.add( "alpha", value=3.5, vary=False, min=0.0, max=5.0 )
    elif target_type == "fluorinated":
        params = Parameters()
        params.add( "beam",      value=0.12, vary=False )
        params.add( "strag",     value=1,    vary=False, min=0.9,  max=1.1   )
        params.add( "n_backing", value=3.0,  vary=True,  min=0.0,  max=10.0  )
        params.add( "n_f",       value=1.0,  vary=False )
        # --- 3-layer erf params (commented out) ---
        params.add( "width1",    value=8.0,  vary=True,  min=1.0, max=25.0 )
        params.add( "width2",    value=10.0, vary=True,  min=0.0, max=35.0 )
        params.add( "width3",    value=20.0, vary=True,  min=0.0, max=40.0 )
        params.add( "norm1",     value=0.3,  vary=True,  min=0.0, max=1.0  )
        params.add( "norm2",     value=0.1,  vary=True,  min=0.0, max=1.0  )
    elif target_type == "evaporated":
        params = Parameters()
        params.add( "beam",  value=0.12, vary=False )
        params.add( "strag", value=1, vary=False, min=0.9, max=1.1  )
        params.add( "n_backing",  value=0.5, vary=True, min=0.0, max=10.0 )
        params.add( "n_f",   value=1.0, vary=False )
        params.add( "width", value=18.0, vary=True, min=0.0, max=100.0 )

    csv_path = f"Yield_scans/Results/Yield_{target}.csv"

    df = pd.read_csv(csv_path)
    if "Scan" not in df.columns:
        print(f"No 'Scan' column in {csv_path}, skipping")
        continue

    scans = df["Scan"].unique()
    for scan_label in scans:
        df_scan = df[df["Scan"] == scan_label]
        x = df_scan["Energy"].to_numpy()
        y = df_scan["Yield"].to_numpy()
        y_err = df_scan["Yield Error"].to_numpy() if "Yield Error" in df_scan.columns else np.ones_like(y)

        # Select points with E not NaN and E > 300 keV
        mask = (~np.isnan(x)) & (x > 300)
        if not np.any(mask):
            print(f"{target} {scan_label}: no E>300 keV points — skipping fit")
            continue

        x = x[mask]
        y = y[mask]
        y_err = y_err[mask]

        # Convert to CM frame
        x = x * 19 / 20.007

        # Replace zero errors if present
        if np.any(y_err == 0):
            nz = y_err[y_err > 0]
            rep = np.median(nz) if nz.size else 1.0
            y_err[y_err == 0] = rep

        # Select efficiency
        eff = eff_far

        # Sort the x, y, y_err arrays by x
        sorted_indices = np.argsort(x)
        x = x[sorted_indices]
        y = y[sorted_indices]
        y_err = y_err[sorted_indices]

        # Iteratively find f_bias so that reduced chi2 = 1 after the fit
        y_err_orig = y_err.copy()
        bias_mask  = ((x < 320) | (x > 345))

        try:
            f_bias   = 0.35
            tol      = 0.02
            max_iter = 10
            for it in range(max_iter):
                bias      = np.mean(y[bias_mask]) * f_bias
                y_err_fit = y_err_orig + bias
                out = minimize( chi2, params, args=(x, y, y_err_fit, eff, target_type, backing_type), max_nfev=5000 )
                redchi = out.redchi if (out.nfree > 0 and np.isfinite(out.redchi)) else float('nan')
                if np.isnan(redchi) or abs(redchi - 1.0) < tol:
                    break
                f_bias *= min(np.sqrt(redchi), 3.0)
            print(f"\nFit results for {target} {scan_label}:")
            for name, p in out.params.items():
                stderr = p.stderr if p.stderr is not None else float('nan')
                print(f"  {name}: {p.value:.6g} +/- {stderr:.6g} (vary={p.vary})")
            print(f"  → redchi2 = {redchi:.4f}, f_bias = {f_bias:.4f} (converged in {it+1} iteration(s))")

            # --- Optional MCMC posterior sampling for corner plots + asymmetric errors ---
            if RUN_MCMC:
                _silent = True
                free_params = [n for n, p in out.params.items() if p.vary]
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
                    flat = out_mcmc.flatchain[free_params].to_numpy()
                    print(f"\nMCMC credible intervals ({target} {scan_label}):")
                    for i, name in enumerate(free_params):
                        q16, q50, q84 = np.percentile(flat[:, i], [16, 50, 84])
                        print(f"  {name}: {q50:.4g}  +{q84-q50:.4g} / -{q50-q16:.4g}")
                    truths = [out.params[n].value for n in free_params]
                    fig_c = corner.corner(flat, labels=free_params,
                                          quantiles=[0.16, 0.5, 0.84],
                                          show_titles=True, title_fmt='.3g',
                                          truths=truths)
                    cname = "".join(c if (c.isalnum() or c in ('_','-')) else '_'
                                    for c in f"{target}_{scan_label}_{ENERGY_TAG}_corner")
                    fig_c.savefig(os.path.join(results_dir, f"{cname}.png"),
                                  dpi=150, bbox_inches='tight')
                    plt.close(fig_c)
                    print(f"Corner plot saved: {cname}.png")
                except Exception as e_mcmc:
                    _silent = False
                    print(f"MCMC failed for {target} {scan_label}: {e_mcmc}")

            # Collect parameters for implanted targets (include parameter errors)
            if target_type == 'implanted':
                def val_and_err(pname):
                    if pname in out.params:
                        p = out.params[pname]
                        v = p.value if p.value is not None else float('nan')
                        e = p.stderr if p.stderr is not None else float('nan')
                    else:
                        v, e = float('nan'), float('nan')
                    return float(v), float(e)

                n_backing, n_backing_err = val_and_err('n_backing')
                mean, mean_err = val_and_err('mean')
                std, std_err = val_and_err('std')
                alpha, alpha_err = val_and_err('alpha')
                chi2sqrt = out.chisqr

                fit_results.append({
                    'target': target,
                    'scan': scan_label,
                    'energy_keV': 340,
                    'n_backing': n_backing,
                    'n_backing_error': n_backing_err,
                    'mean': mean,
                    'mean_error': mean_err,
                    'std': std,
                    'std_error': std_err,
                    'alpha': alpha,
                    'alpha_error': alpha_err,
                    'chi2': chi2sqrt
                })
            # Plot data and model
            try:
                fig, ax = plt.subplots(figsize=(8,6))
                ax.errorbar(x, y, yerr=y_err_fit, fmt='o', label='Data', color='black', capsize=4)
                grid = np.linspace(np.min(x)*0.98, np.max(x)*1.02, 300)
                y_mod = model(grid, out.params, target_type, backing_type) / q_e / 1e6 * eff
                ax.set_title( f"{target} {scan_label}" )
                ax.plot(grid, y_mod, ls='dashed', lw=2, color='deeppink', label='Model')
                ax.set_xlabel(r'$E_p^{\mathrm{cm}}$ [keV]')
                ax.set_ylabel(r'Yield [counts/$\mu$C]')
                ax.set_ylim(0, None)
                ax.grid()

                # # Annotate fit parameters: n_backing (black) and profile params (royalblue)
                # nb = out.params['n_backing'].value
                # nb_err = out.params['n_backing'].stderr if out.params['n_backing'].stderr is not None else float('nan')
                # ax.annotate(fr"$n_{{{backing_type}}}/n_{{F}}$: {nb:.2f} $\pm$ {nb_err:.2f}", xy=(0.95, 0.95), xycoords='axes fraction',
                #             color='black', ha='right', va='top')
                # if target_type == 'implanted':
                #     m = out.params['mean'].value
                #     s = out.params['std'].value
                #     ax.annotate(fr"$\mu$ = {m:.2f} $\pm$ {out.params['mean'].stderr:.2f}", xy=(0.95, 0.88), xycoords='axes fraction',
                #                 color='royalblue', ha='right', va='top')
                #     ax.annotate(fr"$\sigma$ = {s:.2f} $\pm$ {out.params['std'].stderr:.2f}", xy=(0.95, 0.81), xycoords='axes fraction',
                #                 color='royalblue', ha='right', va='top')
                #     # ax.annotate(fr"$\alpha$ = {out.params['alpha'].value:.2f} $\pm$ {out.params['alpha'].stderr:.2f}", xy=(0.95, 0.74), xycoords='axes fraction',
                #     #             color='royalblue', ha='right', va='top')
                #     # ax.annotate(fr"dead layer = {out.params['dead_layer'].value:.2f} $\pm$ {out.params['dead_layer'].stderr:.2f}", xy=(0.95, 0.67), xycoords='axes fraction',
                #     #             color='royalblue', ha='right', va='top')
                # else:
                #     w = out.params.get('width', None)
                #     if w is not None:
                #         ax.annotate(fr"width = {w.value:.2f} $\pm$ {w.stderr:.2f}", xy=(0.95, 0.88), xycoords='axes fraction',
                #                     color='royalblue', ha='right', va='top')

                ax2 = plt.twinx()
                grid_profile = np.linspace( -10, 60, 1000 )
                p = []
                for de in grid_profile:
                    p.append( profile( de, out.params, target_type ) )
                p = np.array( p )
                grid_profile += 323.9
                ax2.plot( grid_profile, p, ls='solid', lw=2, label='Profile', color='royalblue' )
                ax2.fill_between(grid_profile, p, 0, facecolor='None', edgecolor='royalblue', hatch='///////', linewidth=0)
                ax2.set_yticks([])
                ax2.set_ylabel('')
                ax2.set_xlim(310, 360)
                ax2.set_ylim(0, 2)
                safe_name = "".join(c if (c.isalnum() or c in ('_','-')) else '_' for c in f"{target}_{scan_label}_340")
                outpath = os.path.join(results_dir, f"{safe_name}.png")
                fig.savefig(outpath, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"Saved plot: {outpath}")
            except Exception as e:
                print(f"Could not generate plot for {target} {scan_label}: {e}")
        except Exception as e:
            print(f"Fit failed for {target} {scan_label}: {e}")
# After processing all targets/scans, save collected fit parameters
if len(fit_results) > 0:
    results_df = pd.DataFrame(fit_results)
    results_df.to_csv(os.path.join(results_dir, "fit_params_340.csv"), index=False)
    print("Fit results saved to fit_params_340.csv")
else:
    print("No fit results were collected; nothing to save.")

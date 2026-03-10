import argparse
import numpy as np
from Simulations.SRIM import SRIM

# Load SRIM stopping files (paths relative to workspace)
H_in_F = SRIM("Simulations/stopping/H_in_F.stop")
H_in_Ta = SRIM("Simulations/stopping/H_in_Ta.stop")

def effective_stopping_Ta(energy_keV, n_inactive=1, n_active=5):
    stoichiometry = n_inactive / n_active
    return H_in_F.eval(energy_keV) + stoichiometry * H_in_Ta.eval(energy_keV)

def reaction_yield_simple(energy_keV, cross_section_barn, n_ta=5, n_f=1):
    """Return reaction yield per projectile at a given lab energy (keV).

    Parameters
    - energy_keV: projectile energy in keV
    - cross_section_barn: cross section in barn
    - n_ta, n_f: stoichiometry parameters used for effective stopping

    Returns
    - yield (dimensionless) following the same unit-conversions used in the notebook
    """
    cross = cross_section_barn * 1e-24
    stop = effective_stopping_Ta(energy_keV, n_ta, n_f) * 1e-15 * 1e-3
    return cross / stop

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Compute minimal reaction yield at a given energy")
    p.add_argument("energy", type=float, help="Energy in keV")
    p.add_argument("cross", type=float, help="Cross section in barn")
    args = p.parse_args()
    y = reaction_yield_simple(args.energy, args.cross)
    print(f"Reaction yield at {args.energy} keV for {args.cross} barn: {y:.6e} (per projectile)")

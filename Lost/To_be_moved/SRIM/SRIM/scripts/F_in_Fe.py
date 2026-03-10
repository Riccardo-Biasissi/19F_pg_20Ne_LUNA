import os
import sys

from srim import Ion, Layer, Target, TRIM

# Get arguments
idx = int(sys.argv[1])
energy = float(sys.argv[2])

# Construct an ion
ion = Ion('F', energy=energy * 1e3)

# Construct a layer
layer = Layer({'Fe': { 'stoich': 1, 'E_d': 25.0, 'lattice': 0.0, 'surface': 3.0}}, 
                density=7.87, width=5e4)

# Construct a target
target = Target([layer])

# Initialize a TRIM calculation
trim = TRIM(target, ion, number_ions=1000, calculation=1, ranges=True)

# Specify the directory of SRIM.exe
srim_executable_directory = '/tmp/srim'
results = trim.run(srim_executable_directory)

os.makedirs('/tmp/output', exist_ok=True)
os.system(f'cp "/tmp/srim/SRIM Outputs/RANGE_3D.txt" /tmp/output/RANGE_F_IN_FE_ENERGY_{energy}KEV_{idx}.txt')
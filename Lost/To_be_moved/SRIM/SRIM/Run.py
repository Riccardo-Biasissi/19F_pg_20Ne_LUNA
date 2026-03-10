import os

# Energy of the ion in keV
e = 30

for i in range( 100 ):
    os.system(f'/data0/biasissi/LUNA/19F+p_g+20Ne/Scripts/To_be_moved/SRIM/SRIM/./run.sh {i} {e}')
import re
import numpy as np

class SRIM:

    def __init__(self, filename):
        self.filename = filename
        self.data = []
        self.parse()

    def parse(self):
        
        with open(self.filename, 'r') as file:
            lines = file.readlines()

        table_start_pattern = re.compile(r'\s*Energy\s+Elec\.\s+Nuclear\s+Range\s+Straggling\s+Straggling')
        table_stop_pattern = re.compile(r'\s*Multiply Stopping by')

        table_start = None
        table_stop = None
        
        for i, line in enumerate(lines):

            table_match = table_start_pattern.match(line)
            if table_match:
                table_start = i + 2

            table_match = table_stop_pattern.match(line)
            if table_match:
                table_stop = i - 2
        
        table = lines[table_start:table_stop]

        array = np.zeros((len(table), 6))
        for i, line in enumerate(table):

            line = line.replace(',', '.')
            line = line.split()

            if( line[1] == 'keV'):
                array[i, 0] = float(line[0])
            elif( line[1] == 'MeV'):
                array[i, 0] = float(line[0]) * 1000

            array[i, 1] = float(line[2])
            array[i, 2] = float(line[3])

            if( line[5] == 'A'):
                array[i, 3] = float(line[4])
            elif( line[5] == 'um'):
                array[i, 3] = float(line[4]) * 10000

            if( line[7] == 'A'):
                array[i, 4] = float(line[6])
            elif( line[7] == 'um'):
                array[i, 4] = float(line[6]) * 10000

            if( line[9] == 'A'):
                array[i, 5] = float(line[8])
            elif( line[9] == 'um'):
                array[i, 5] = float(line[8]) * 10000

        self.data = array

    def get_data(self):
        return self.data

    def get_energy(self):
        return self.data[:,0]

    def get_dedx(self):
        return self.data[:,1]
    
    def get_dedx_nuclear(self):
        return self.data[:,2]

    def get_projected_range(self):
        return self.data[:,3]

    def get_longitudinal_straggle(self):
        return self.data[:,4]
    
    def get_lateral_straggle(self):
        return self.data[:,5]
    
    def eval( self, energy ):
        return np.interp(energy, self.get_energy(), self.get_dedx())
    
    def eval_nuclear( self, energy ):
        return np.interp(energy, self.get_energy(), self.get_dedx_nuclear())
    
    def eval_range( self, energy ):
        return np.interp(energy, self.get_energy(), self.get_projected_range())
    
    def eval_longitudinal_straggle( self, energy ):
        return np.interp(energy, self.get_energy(), self.get_longitudinal_straggle())
    
    def eval_lateral_straggle( self, energy ):
        return np.interp(energy, self.get_energy(), self.get_lateral_straggle())
    
    def eval_total_straggle( self, energy ):
        return np.interp(energy, self.get_energy(), self.get_dedx() + self.get_dedx_nuclear())
    

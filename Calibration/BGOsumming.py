import os
from typing import List, Optional
import numpy as np


def bgo_summing(custom_run: List[int],
				main_calib_run: int,
				coincidence_window: int = 100,
				custom_calib: int = 0,
				max_timestamp: Optional[int] = None,
				root_dir: str = "/data0/biasissi/LUNA/19F+p_g+20Ne/Data/ROOT/",
				processed_dir: str = "/data0/biasissi/LUNA/19F+p_g+20Ne/Data/PROCESSED/") -> None:
	"""
	Run the external BGO summing binary for an explicit list of runs.

	This simplified version removes any dependency on `secondary_calib_run`, `target`, and
	`pre_calib`. You must provide `custom_run` (non-empty iterable of run numbers). The
	calibration selection uses only `main_calib_run` unless `custom_calib != 0`, in which
	case a test calibration file is used.
	"""
	if custom_run is None or len(custom_run) == 0:
		raise ValueError("'custom_run' must be provided as a non-empty list of run numbers when not using a logbook.")

	def _energy_calib_option_for(run: int) -> str:
		if custom_calib != 0:
			return "--energy-calibration-file /data0/biasissi/LUNA/19F+p_g+20Ne/test_calib.txt "
		return f"--energy-calibration-file /data0/biasissi/LUNA/19F+p_g+20Ne/Calibration/Params/calibration_run{main_calib_run}.txt "

	for run in custom_run:
		print(f"Analysing Run{run}")

		cmd = "/data0/biasissi/LUNA/19F+p_g+20Ne/Codes/luna-bgo-summing/build/BGOsumming "
		cmd += f"--input-root-file {root_dir}/run{run}.root "
		cmd += f"--coincidence-time-window {coincidence_window} "
		cmd += _energy_calib_option_for(run)
		cmd += "--charge-channel 7 "
		cmd += "--no-charge "
		cmd += "--pulser-channel 0 "
		# cmd += "--no-pulser "
		if max_timestamp:
			cmd += f"--cut-after-timestamp {max_timestamp} "
		cmd += "--write-coincidence-event-tree "
		cmd += "--include-raw-in-coincidence-event-tree "
		cmd += f"--output-file {processed_dir}/run{run}.root"

		print(f"Running: {cmd}")
		os.system(cmd)

# Import only bgo_summing function when saying from BGOsumming import *
__all__ = ["bgo_summing"]


# RUNS = np.arange(3260, 3263+1)
RUNS = [668]
# RUNS = [1516]
for run in RUNS:
	bgo_summing(custom_run=[run], main_calib_run=668, coincidence_window=1000)




#####


# I need you to open the RUNS.root file. Inside it you will find a ttree called V1724. I want you to renominate it as DataR and save the file as RUNS_renamed.root
# Make sure to keep all the data inside the ttree unchanged, just change its name.
# You can use ROOT or any other library you prefer to do this task.
# import ROOT
# def rename_ttree(input_file: str, output_file: str, old_tree_name: str, new_tree_name: str) -> None:
# 	# Open the input ROOT file
# 	input_root = ROOT.TFile.Open(input_file, "READ")
	
# 	# Get the old TTree
# 	old_tree = input_root.Get(old_tree_name)
# 	if not old_tree:
# 		raise ValueError(f"TTree '{old_tree_name}' not found in file '{input_file}'")
	
# 	# Create the output ROOT file
# 	output_root = ROOT.TFile.Open(output_file, "RECREATE")
	
# 	# Clone the old TTree to the new TTree with the new name
# 	new_tree = old_tree.CloneTree()
# 	new_tree.SetName(new_tree_name)
	
# 	# Write the new TTree to the output file
# 	output_root.cd()
# 	new_tree.Write()
	
# 	# Close both files
# 	input_root.Close()
# 	output_root.Close()

# Example usage:
# rename_ttree("/data0/biasissi/LUNA/19F+p_g+20Ne/Data/ROOT/run1090.root", "/data0/biasissi/LUNA/19F+p_g+20Ne/Data/ROOT/run1090_renamed.root", "V1724", "DataR")
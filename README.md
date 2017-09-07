# false-alarm-reduction
Code for building a model to reduce false alarms in the intensive care unit. 

1. Download and install the following packages: `wfdb`, `csv`, `numpy`, `scipy`, `matplotlib`.
2. Download data and annotations (Challenge dataset used here: https://physionet.org/challenge/2015/training.zip) and modify the `data_path` and `ann_path` variables accordingly in `parameters.py`. 
3. To run
..* baseline algorithm, update `write_filename` in `parameters.py` and run `pipeline.py`. 
..* DTW algorithm, update the `matrix_filename` and `distances_filename` variables in `parameters.py` and run `dtw.py`.
..* baseline and DTW algorithms together, run `combine_baseline_dtw.py`. 



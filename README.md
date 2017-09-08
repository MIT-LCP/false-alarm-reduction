# false-alarm-reduction
Code for building a model to reduce false alarms in the intensive care unit. 

1. Download and install the following packages: `wfdb`, `numpy`, `scipy`, `matplotlib`, `sklearn`, `fastdtw`, `spectrum`, `peakutils`.
2. Download data and annotations (Challenge dataset used here: https://physionet.org/challenge/2015/training.zip TODO: where to download annotations???) and modify the `data_path` and `ann_path` variables accordingly in `parameters.py` to point to the data files and annotation files, respectively. 
3. To run
  * baseline algorithm, update `write_filename` in `parameters.py` to be the filename to output the results of the baseline algorithm. Then run `pipeline.py`. 
  * DTW algorithm on the alarm signal, update the `matrix_filename` and `distances_filename` variables in `parameters.py` to be the filenames to output the final confusion matrix and corresponding distance results, respectively. Then, run `dtw.py`.
  * DTW algorithm beat-by-beat (bank), update `output_path_bank` in `parameters.py` to be the path to the folder desired for ventricular beat annotations via standard beat comparisons. Then run `ventricular_beat_bank.py`. In `baseline_algorithm.py`, comment out lines 916-917 and uncomment line 918. Make sure that the `output_path` on line 869 in the `read_ventricular_beat_annotations` function is set to `parameters.output_path_bank` in `baseline_algorithm.py`.
  * DTW algorithm beat-by-beat (standard deviation), update `output_path_std` in `parameters.py` to be the path to the folder desired for ventricular beat annotations via standard deviation calculations. Then run `ventricular_beat_std.py`. In `baseline_algorithm.py`, comment out lines 916-917 and uncomment line 918. Make sure that the `output_path` on line 869 in the `read_ventricular_beat_annotations` function is set to `parameters.output_path_std` in `baseline_algorithm.py`.
4. To run using a different QRS detector (JQRS instead of GQRS), change `ecg_ann_type` to `jqrs`. 


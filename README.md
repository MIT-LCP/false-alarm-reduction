# false-alarm-reduction
Code for building a model to reduce false alarms in the intensive care unit.


## Detailed Installation (Ubuntu 16.04)

### Install Virtual Environment (Python 2.7)

We recommend installing the package in a virtual environment. If not using a virtual environment, skip ahead to the `pip install pip --upgrade` step.

First, install pip and virtualenv as follows:

```
sudo apt-get install python-pip python-dev python-virtualenv
```

Create a virtual environment

```
virtualenv --system-site-packages TARGET_DIRECTORY
```

Where TARGET_DIRECTORY is the desired location of the virtual environment. Here, we assume it is ``~/false-alarm-reduction`.

Activate the virtual environment

```
source ~/false-alarm-reduction/bin/activate
```

Now you should be working in the virtual environment. Verify pip is installed:

```
(false-alarm-reduction)$ easy_install -U pip
```

Upgrade pip.

```
(false-alarm-reduction)$ pip install pip --upgrade
```

Now install all the necessary packages using the requirements file (for reference, these are: `wfdb`, `numpy`, `scipy`, `matplotlib`, `sklearn`, `fastdtw`, `spectrum`, `peakutils`).

```
pip install -r requirements.txt
```

### Download data

Two convenience scripts are provided to download the data. Run these from the main folder as follows:

```
sh download/download_annotations.sh
sh download/download_data.sh
```

These scripts download the following data:

* training.zip https://physionet.org/challenge/2015/training.zip
* ann_gqrs0.zip https://www.dropbox.com/sh/hv4uat0ihwlygq8/AABvdXbSGZi3COPG-O_-nBGxa/ann_gqrs0.zip?dl=1
* ann_gqrs1.zip https://www.dropbox.com/sh/hv4uat0ihwlygq8/AAAAb14a_NN8iKojXEoInXCGa/ann_gqrs1.zip?dl=1
* ann_wabp.zip  https://www.dropbox.com/sh/hv4uat0ihwlygq8/AAALSmteHaL0gQovwXj8CXV4a/ann_wabp.zip?dl=1
* ann_wpleth.zip https://www.dropbox.com/sh/hv4uat0ihwlygq8/AAAko1RNvgmdhWF7lNux-Ob3a/ann_wpleth.zip?dl=1

Afterward, you should have the following directory structure:

* `annotations` subfolder with all annotations (*.gqrs0, *.gqrs1, *.wabp, *.wpleth)
* `data/training` subfolder with all data (*.mat and *.hea) and a RECORDS file

## Brief instructions for other OS

1. Download and install the following packages: `wfdb`, `numpy`, `scipy`, `matplotlib`, `sklearn`, `fastdtw`, `spectrum`, `peakutils`.
2. Download data and annotations (Challenge dataset used here: https://physionet.org/challenge/2015/training.zip TODO: where to download annotations???) and modify the `data_path` and `ann_path` variables accordingly in `parameters.py` to point to the data files and annotation files, respectively.
3. To run
  * baseline algorithm, update `write_filename` in `parameters.py` to be the filename to output the results of the baseline algorithm. Then run `pipeline.py`.
  * DTW algorithm on the alarm signal, update the `matrix_filename` and `distances_filename` variables in `parameters.py` to be the filenames to output the final confusion matrix and corresponding distance results, respectively. Then, run `dtw.py`.
  * DTW algorithm beat-by-beat (bank), update `output_path_bank` in `parameters.py` to be the path to the folder desired for ventricular beat annotations via standard beat comparisons. Then run `ventricular_beat_bank.py`. In `baseline_algorithm.py`, comment out lines 916-917 and uncomment line 918. Make sure that the `output_path` on line 869 in the `read_ventricular_beat_annotations` function is set to `parameters.output_path_bank` in `baseline_algorithm.py`.
  * DTW algorithm beat-by-beat (standard deviation), update `output_path_std` in `parameters.py` to be the path to the folder desired for ventricular beat annotations via standard deviation calculations. Then run `ventricular_beat_std.py`. In `baseline_algorithm.py`, comment out lines 916-917 and uncomment line 918. Make sure that the `output_path` on line 869 in the `read_ventricular_beat_annotations` function is set to `parameters.output_path_std` in `baseline_algorithm.py`.
4. To run using a different QRS detector (JQRS instead of GQRS), change `ecg_ann_type` to `jqrs`.

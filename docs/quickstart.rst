Quickstart
==========

Install
-------

::

    $ pip install false-alarm-reduction

See :doc:`installation <install>` document for more information.


Acquire data
------------

The easiest way to understand what this package does is to evaluate it on
physiologic waveforms. The 2015 PhysioNet/Computing in Cardiology Challenge
focused on false alarm reduction and provides a useful dataset to work with.

To download this dataset, run the download shell script:

.. code-block:: bash

  bash download/download_data.sh

This will download data into the ``data`` subfolder using ``wget`` and decompress
the individual files.

See the :doc:`Challenge 2016 data <challenge2016data>` document for more
information on the dataset.

More detail on the dataset can be found on the `Challenge 2016 website`__.

.. _challenge2016: https://physionet.org/challenge/2016/

__ challenge2016_

Acquire R-peak annotations
--------------------------

R-peak annotations indicate where in the electrocardiogram (ECG) heart beat
cycle the "R" peak is estimated to be. You can read about the different waves
in the ECG from `ecgpedia basics`__.

.. _ecgpediabasics: http://en.ecgpedia.org/index.php?title=Basics#The_different_ECG_waves

__ ecgpediabasics_

R-peak annotations can be acquired in two ways: (1) downloading them directly,
or (2) generating them from the data.

Downloading R-peak annotations
------------------------------

Run the bash script to download annotations:

.. code-block:: bash

  bash download/download_annotations.sh

Generating R-peak annotations
-----------------------------

Annotations can be regenerated from the data itself. Two annotation software
tools are required: GQRS and WABP. GQRS uses the ECG to identify R-peaks, while
WABP uses pulsatile waveforms such as the arterial blood pressure (ABP) waveform
or the photoplethysmogram (PPG) to identify these peaks.

First, install the WFDB toolbox from `PhysioNet`__.

.. _wfdb: https://physionet.org/tools

__ wfdb_

Then, run the annotation script. This iterates through all data records and generates annotations.

.. code-block:: bash

  cd annotations
  bash annotate_data.sh


Running the code
----------------

Baseline algorithm
~~~~~~~~~~~~~~~~~~

You can run the main "baseline" algorithm using:

``python pipeline.py``

This will run through all the data files and associated annotation files
to detect/flag false alarms. The output files are written at the end of
the algorithm to the current directory. For the baseline algorithm, this
is by default ``results.json``.

Note: To run using a different QRS detector (e.g. JQRS instead of GQRS),
change ``ecg_ann_type``, e.g. ``ecg_ann_type = "jqrs"``. See the
``matlab/`` subfolder for code to generate JQRS (note this code is
untested!).

DTW time warping
~~~~~~~~~~~~~~~~

To run the DTW algorithm on the alarm signal, update the
``matrix_filename`` and ``distances_filename`` variables in
``parameters.py`` to be the filenames to output the final confusion
matrix and corresponding distance results, respectively. Then, call the
algorithm as ``python dtw.py``.

(Experimental) Using ventricular/normal beat banks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  DTW algorithm beat-by-beat (bank): update ``output_path_bank`` in
   ``parameters.py`` to be the path to the folder desired for
   ventricular beat annotations via standard beat comparisons. Then run
   ``ventricular_beat_bank.py``. In ``baseline_algorithm.py``, comment
   out lines 916-917 and uncomment line 918. Make sure that the
   ``output_path`` on line 869 in the
   ``read_ventricular_beat_annotations`` function is set to
   ``parameters.output_path_bank`` in ``baseline_algorithm.py``.
-  DTW algorithm beat-by-beat (standard deviation), update
   ``output_path_std`` in ``parameters.py`` to be the path to the folder
   desired for ventricular beat annotations via standard deviation
   calculations. Then run ``ventricular_beat_std.py``. In
   ``baseline_algorithm.py``, comment out lines 916-917 and uncomment
   line 918. Make sure that the ``output_path`` on line 869 in the
   ``read_ventricular_beat_annotations`` function is set to
   ``parameters.output_path_std`` in ``baseline_algorithm.py``.

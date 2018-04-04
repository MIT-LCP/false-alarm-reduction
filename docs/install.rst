Install pyfar
=============

Brief instructions
------------------

You can install pyfar with ``conda``, with ``pip``, or from source.

Conda
~~~~~

To install the latest version of pyfar from the
`conda-forge <https://conda-forge.github.io/>`_, run
`conda <https://www.anaconda.com/downloads>`_::

    conda install pyfar -c conda-forge

Pip
~~~

Or install pyfar with ``pip``::

    pip install pyfar --upgrade

Source
~~~~~~

To install pyfar from source, clone the repository from `github
<https://github.com/MIT-LCP/false-alarm-reduction>`_::

    git clone https://github.com/MIT-LCP/false-alarm-reduction.git
    cd false-alarm-reduction
    python setup.py install


Detailed instructions
---------------------

These instructions were tested on Ubuntu 16.04.

Install Virtual Environment (Python 2.7)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We recommend installing the package in a virtual environment. If not
using a virtual environment, skip ahead to the
``pip install pip --upgrade`` step.

First, install pip and virtualenv as follows:

::

    sudo apt-get install python-pip python-dev python-virtualenv

Create a virtual environment

::

    virtualenv --system-site-packages TARGET_DIRECTORY

Where TARGET\_DIRECTORY is the desired location of the virtual
environment. Here, we assume it is \`\ ``~/false-alarm-reduction``.

Activate the virtual environment

::

    source ~/false-alarm-reduction/bin/activate

Now you should be working in the virtual environment. Verify pip is
installed:

::

    (false-alarm-reduction)$ easy_install -U pip

Upgrade pip.

::

    (false-alarm-reduction)$ pip install pip --upgrade

Now install all the necessary packages using the requirements file (for
reference, these are: ``wfdb``, ``numpy``, ``scipy``, ``matplotlib``,
``sklearn``, ``fastdtw``, ``spectrum``, ``peakutils``).

::

    pip install -r requirements.txt

Download data
~~~~~~~~~~~~~

Two convenience scripts are provided to download the data. Run these
from the main folder as follows:

::

    sh download/download_annotations.sh
    sh download/download_data.sh

These scripts download the following data:

-  training.zip https://physionet.org/challenge/2015/training.zip
-  ann\_gqrs0.zip
   https://www.dropbox.com/sh/hv4uat0ihwlygq8/AABvdXbSGZi3COPG-O\_-nBGxa/ann\_gqrs0.zip?dl=1
-  ann\_gqrs1.zip
   https://www.dropbox.com/sh/hv4uat0ihwlygq8/AAAAb14a\_NN8iKojXEoInXCGa/ann\_gqrs1.zip?dl=1
-  ann\_wabp.zip
   https://www.dropbox.com/sh/hv4uat0ihwlygq8/AAALSmteHaL0gQovwXj8CXV4a/ann\_wabp.zip?dl=1
-  ann\_wpleth.zip
   https://www.dropbox.com/sh/hv4uat0ihwlygq8/AAAko1RNvgmdhWF7lNux-Ob3a/ann\_wpleth.zip?dl=1

Afterward, you should have the following directory structure:

-  ``annotations`` subfolder with all annotations (*.gqrs0, *.gqrs1,
   *.wabp, *.wpleth)
-  ``data/training`` subfolder with all data (*.mat and *.hea) and a
   RECORDS file

Brief instructions for other OS
-------------------------------

1. Download and install the following packages: ``wfdb``, ``numpy``,
   ``scipy``, ``matplotlib``, ``sklearn``, ``fastdtw``, ``spectrum``,
   ``peakutils``.
2. Download data and annotations

-  training.zip https://physionet.org/challenge/2015/training.zip
-  ann\_gqrs0.zip
   https://www.dropbox.com/sh/hv4uat0ihwlygq8/AABvdXbSGZi3COPG-O\_-nBGxa/ann\_gqrs0.zip?dl=1
-  ann\_gqrs1.zip
   https://www.dropbox.com/sh/hv4uat0ihwlygq8/AAAAb14a\_NN8iKojXEoInXCGa/ann\_gqrs1.zip?dl=1
-  ann\_wabp.zip
   https://www.dropbox.com/sh/hv4uat0ihwlygq8/AAALSmteHaL0gQovwXj8CXV4a/ann\_wabp.zip?dl=1
-  ann\_wpleth.zip
   https://www.dropbox.com/sh/hv4uat0ihwlygq8/AAAko1RNvgmdhWF7lNux-Ob3a/ann\_wpleth.zip?dl=1

3. Data should be unzipped into ``data/`` (ultimately the files will be
   in ``data/training/``)
4. Annotations should be unzipped into ``annotations/``

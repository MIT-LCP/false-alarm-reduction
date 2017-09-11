#!/bin/bash
# downloads the data to the data subfolder
# run this script from the root folder of the cloned repository

mkdir -p data

cd data
wget -O training.zip https://physionet.org/challenge/2015/training.zip
unzip training.zip
cd ..

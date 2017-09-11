#!/bin/bash
# downloads the annotations (R peak detections) to the annotations subfolder
# run this script from the root folder of the cloned repository

mkdir -p annotations

wget -O annotations/ann_gqrs0.zip https://www.dropbox.com/sh/hv4uat0ihwlygq8/AABvdXbSGZi3COPG-O_-nBGxa/ann_gqrs0.zip?dl=1
wget -O annotations/ann_gqrs1.zip https://www.dropbox.com/sh/hv4uat0ihwlygq8/AAAAb14a_NN8iKojXEoInXCGa/ann_gqrs1.zip?dl=1
wget -O annotations/ann_wabp.zip https://www.dropbox.com/sh/hv4uat0ihwlygq8/AAALSmteHaL0gQovwXj8CXV4a/ann_wabp.zip?dl=1
wget -O annotations/ann_wpleth.zip https://www.dropbox.com/sh/hv4uat0ihwlygq8/AAAko1RNvgmdhWF7lNux-Ob3a/ann_wpleth.zip?dl=1

cd annotations
unzip ann_gqrs0.zip
unzip ann_gqrs1.zip
unzip ann_wabp.zip
unzip ann_wpleth.zip
cd ..

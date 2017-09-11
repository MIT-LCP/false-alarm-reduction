---
title: 'False alarm reduction in the intensive care unit'
tags:
  - false alarm reduction
  - signal processing
  - intensive care unit
  - dynamic time warping
  - arrhythmia
authors:
 - name: Andrea S. Li
   orcid: 0000-0001-8419-5527
   affiliation: 1
 - name: Alistair E. W. Johnson
   orcid: 0000-0002-8735-3014
   affiliation: 1
 - name: Roger G. Mark
   orcid: 0000-0002-6318-2978
   affiliation: 1
affiliations:
 - name: Massachusetts Institute of Technology
   index: 1
date: 11 September 2017
bibliography: paper.bib
---

# Summary

This is an algorithm for reducing the number of false arrhythmia alarms reported by intensive care unit monitors.
Research has shown that only 17\% of alarms in the intensive care unit (ICU) are clinically relevant [@siebig].
The high false arrhythmia alarm rate has severe implications such as disruption of patient care, caregiver alarm fatigue, and desensitization from clinical staff to real life-threatening alarms  [@imhoff].
A method to reduce the false alarm rate would therefore greatly benefit patients as well as nurses in their ability to provide care. We here develop and describe a robust false arrhythmia alarm reduction system for use in the ICU.
We utilize the PhysioNet/Computing in Cardiology (CinC) Challenge 2015 dataset for development and validation of our approach [@challenge].
Building off of work previously described in the literature [@plesinger], we make use of signal processing and machine learning techniques to identify true and false alarms for five arrhythmia types.
This baseline algorithm alone is able to perform remarkably well, with a sensitivity of 0.908, a specificity of 0.838, and a PhysioNet/CinC challenge score of 0.756 [@challenge].
We additionally explore dynamic time warping techniques on both the entire alarm signal as well as on a beat-by-beat basis in an effort to improve performance of ventricular tachycardia, which has in the literature been one of the hardest arrhythmias to classify. Such an algorithm with strong performance and efficiency could potentially be translated for use in the ICU to promote overall patient care and recovery.
The software is published to zenodo with DOI 'http://dx.doi.org/10.5281/zenodo.889036' [@fareduction].

# References

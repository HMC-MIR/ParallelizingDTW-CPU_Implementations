# SegmentalDTW

This repository contains the code for the ICASSP 2021 paper "Segmental DTW: A Parallelizable Alternative to Dynamic Time Warping."

The goal of this project is to globally align two feature sequences with a parallelizable algorithm.

You can find the paper [here](http://pages.hmc.edu/ttsai/assets/SegmentalDTW_icassp2021.pdf).

Simply clone the virtual environment and run the jupyter notebooks in order.  Because the second notebook takes a long time to run, you can also run it from the command line with:

`jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.enabled=True --ExecutePreprocessor.timeout=-1 02_Align.ipynb`

The root directories for the Chopin Mazurka dataset and for any output can be changed at the top of each relevant notebook. Please note that the output directory should be consistent across all notebooks.

The AlternateAlign.ipynb notebook contains additional implementations for Segmental DTW using Numba. This notebook does not need to be run, but can be used if the cython implementations fail.

The parallelized, GPU based implementation for Segmental DTW can be found at [this repository](https://github.com/HMC-MIR/SegmentalDTWCUDA).

## Citation

TJ Tsai. "Segmental DTW: A Parallelizable Alternative to Dynamic Time Warping" in Proceedings of the IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), 2021.

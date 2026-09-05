# Introduction
This repository contains the codes to generate constrained surrogates for several types of sistributions: powerlaw, (left-truncated) lognormal, exponential,  (left-truncated) Gaussian, and uniform.

It allows you to

- generate constrained surrogates based on a time series
- perform simple hypothesis tests with constrained surrogates
- reproduce the results of the manuscript Constrained surrogates for arbitrary families of continuous probability distributions, by Jack Murdoch Moore, and Eduardo G. Altmann


# How to use


A tutorial to generate surrogates based on a new or existing time series is given in the [Jupyter notebook](https://jupyter.org/) ['tutorial.ipynb'](https://github.com/JackMurdochMoore/power-law/blob/main/tutorial.ipynb) in the current folder.

In order to reproduce the results of the manuscript, you should run the notebook 'generate-results.ipynb' with the parameters of the manuscript (to generate the results) and the notebook 'make-figures.ipynb' (to generate the figures). Both 'generate-results.ipynb' and 'make-figures.ipynb' are in the folder [reproduce-paper](https://github.com/JackMurdochMoore/power-law/tree/main/reproduce-paper).


# Organization of the repository:

## Folders

- src: contains source code (i.e., the module 'constrained_power_law_surrogates.py')
- time-series: contains the data used in this repository
- reproduce-paper: code, output data and figures that reproduce the results of the manuscript


## Files

- 'requirements.txt': python packages required in the repository.
- 'tutorial.ipynb': A tutorial to generate surrogates based on a new or [existing time series](https://github.com/JackMurdochMoore/power-law/tree/main/time-series).

## References

- "Constrained surrogates for arbitrary families of continuous probability distributions", Jack Murdoch Moore, Gand Eduardo G. Altmann, Constrained surrogates for arbitrary families of continuous probability distributions.

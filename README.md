# Introduction
This repository contains the codes to generate constrained surrogates for several types of distributions: powerlaw, (truncated) lognormal, exponential,  (truncated) Gaussian, and uniform.

It allows you to

- generate constrained surrogates based on a time series
- perform simple hypothesis tests with constrained surrogates
- estimate expected values of statistics using constrained surrogates
- reproduce the results of the manuscript "Constrained surrogates for arbitrary families of continuous probability distributions," by Jack Murdoch Moore and Eduardo G. Altmann


# How to use


A tutorial to generate surrogates based on a new or existing time series is given in the [Jupyter notebook](https://jupyter.org/) ['tutorial.ipynb'](https://github.com/JackMurdochMoore/surrogates-for-distributions/blob/main/tutorial.ipynb) in the current folder.

In order to reproduce the results of the manuscript, you should run the notebooks in the folder [reproduce-paper](https://github.com/JackMurdochMoore/surrogates-for-distributions/tree/main/reproduce-paper).


# Organization of the repository:

## Folders

- src: contains source code (i.e., the module 'constrained_likelihood_surrogates.py')
- data: contains the data used in this repository
- reproduce-paper: code, output data and figures that reproduce the results of the manuscript


## Files

- 'requirements.txt': python packages required in the repository.
- 'tutorial.ipynb': A tutorial to generate surrogates based on new or [existing data](https://github.com/JackMurdochMoore/surrogates-for-distributions/tree/main/data).

## References

- "Constrained surrogates for arbitrary families of continuous probability distributions," Jack Murdoch Moore and Eduardo G. Altmann.

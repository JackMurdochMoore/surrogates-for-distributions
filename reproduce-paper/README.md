# Constrained surrogates for arbitrary families of continuous probability distributions

Code and data associated with the manuscript:

> "Constrained surrogates for arbitrary families of continuous probability distributions"\
> Jack Murdoch Moore and Eduardo G. Altmann

## Jupyter notebooks

Most analyses use two notebooks:

-   `analysis-*.ipynb` notebooks analyse a dataset or perform a
    numerical experiment and save the results.
-   The corresponding `plots-*.ipynb` notebooks use those saved results
    to generate the figures.

The exception is:

-   `analysis-and-plots-cities.ipynb`, which analyses the
    city-population data, saves the results, and generates the figures
    without a separate plotting notebook.

The hypothesis-testing notebooks have two versions:

-   `analysis-hypothesis-testing.ipynb` performs hypothesis tests for
    synthetic data.
-   `analysis-hypothesis-testing-and-computational-time.ipynb` extends
    `analysis-hypothesis-testing.ipynb`. In addition to performing
    hypothesis tests for synthetic data, it records the computational
    cost of different surrogate methods. It also generates each
    surrogate independently from the original input data rather than
    generating each surrogate from the previous surrogate.

### Analysis and plotting notebooks

Reproduces Fig. 5 and Fig. S8:

`analysis-estimates-of-statistics.ipynb` and
`plots-estimates-of-statistics.ipynb` estimate statistics using
synthetic data. Results are stored in `./results/est-stat/`.

Reproduces Fig. 4 and Figs. S1--S7:

`analysis-hypothesis-testing.ipynb` and `plots-hypothesis-testing.ipynb`
perform hypothesis tests for synthetic data. Results are stored in
`./results/hyp-test/`.

Reproduces the computational-time results in the Supplementary Material:

`analysis-hypothesis-testing-and-computational-time.ipynb` and
`plots-hypothesis-testing-and-computational-time.ipynb` perform
hypothesis tests for synthetic data while recording computational costs.
Results are stored in `./results/hyp-test/`.

Reproduces Fig. 8:

`analysis-brain.ipynb` and `plots-brain.ipynb` analyse neuronal data in
`./data/brain/`. Results are stored in `./results/hyp-test/`.

Reproduces Fig. S12 and Table S1:

`analysis-and-plots-cities.ipynb` analyses city-population data in
`./data/cities/`. Results are stored in `./results/hyp-test/`.

Reproduces Fig. 6:

`analysis-fires.ipynb` and `plots-fires.ipynb` analyse fire-size data in
`./data/fires/`. Results are stored in `./results/est-stat/` and
`./results/hyp-test/`.

Reproduces Fig. 7 and Fig. S11:

`analysis-words.ipynb` and `plots-words.ipynb` analyse word-waiting-time
data in `./data/words/`. Results are stored in `./results/hyp-test/`.
Example results for the words data are not included because they
comprise hundreds of individual files.

## Folders

`./src/` contains the main source-code file:
`constrained_likelihood_surrogates.py`

`./data/` contains empirical datasets:

-   `./data/brain/`: neuronal data.
-   `./data/cities/`: city-population data.
-   `./data/fires/`: fire-size data.
-   `./data/words/`: word-waiting-time data.

`./results/` stores results generated from empirical or synthetic data:

-   `./results/est-stat/`: results related to estimation of statistics.
-   `./results/hyp-test/`: results related to hypothesis testing.

`./figures/` stores figures generated from saved results.

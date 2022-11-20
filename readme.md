# Heterogeneous generalized zero-shot domain adaptation
This repository includes codes and scripts for the scientific research project: *Cross Species Zero-Shot Celltype Classifcation Using Single-Cell RNA Data*.

The main part is a python implementation of *CDSPP* (<https://www.sciencedirect.com/science/article/abs/pii/S0031320321005422>) adapted to heterogeneous GZSDA.
Several other functions are connected to the specific task of cross-species cell type annotation.
Both the preprocessed data sets of this research as well as an example for het. GZSDA is provided (*het_GZSDA.ipynb*).

The *main.py* functions runs all possible masked combinations of cells and stores the results in an outputfile, hyper-parameter selection included.


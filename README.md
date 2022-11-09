**cheMLearning** library provides of high-level, user-friendly tools to perform common chemometrics analysis in untargeted metabolomics.
Modelling is done using scikit-learn framework, for machine learning; while visualizations are mainly done with Plotly.

Main modules are: 
- unsupervised_modelling 
- supervised_modelling
- utils

## Installation

```python
pip install cheMLearning
```
within a Jupyter Notebook cell

```python
!pip install cheMLearning
```



# Unsupervised modelling

Unsupervised class works around `Principal Component Analysis`, facilitating the visualization of score and loading plots. Moreover, `Hierarchical Clustering Analysis` can be called on the 
principal components and visualize a score plot where the color code is given by the clusters of the HCA.

Visualizations can be obtained without having to manipulate scores or loadings. However, if one would like to access to these outputs, they can be accessed via class modules.

## Import

```python
from cheMLearning.unsupervised_modelling import unsupervised_class
```

# Supervised modelling

Supervised class includes `Partial Least Squares` and `Random Forest` which can be used for either regression or classification tasks. PLS is computed with the NIPALS algorithm.

In the PLS modelling, tools to compute goodness metrics, i.e. R² and Q², VIPs, are provided. In addition to a static functions to ease feature extraction VIP for a 
given VIP threshold.

For RF modelling, it can be computed the distance matrix for a RF-classifier model. It allows tracking where the observations land in the forest.

Analytical tools for classifiers are also provided by facilitating a classification report.

## Import

```python
from cheMLearning.unsupervised_modelling import supervised_class
```

# Utils

Here, tools to create confusion matrices and ROC curves are provided.
In case one is working with MetaboRank (MR) tool, a function to read MR's output is provided.

## Import

```python
from cheMLearning import utils
```

Change Log
==========

0.0.1 (09/11/2022)
------------------
- First Release
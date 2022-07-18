# annotated_MIND
Python implementation of [Manifold Inference from Neural Dynamics](https://www.biorxiv.org/content/10.1101/418939v2) (MIND).

## About MIND

MIND is an [unsupervised learning](https://en.wikipedia.org/wiki/Unsupervised_learning) algortihm designed to discover low-dimensional manifolds from high-dimensional dynamical/time series data. The method was co-invented by *Ryan Low* and *Sam Lewallen* to analyze neural recording data and is brilliantly explained in their [BioRxiv publication](https://www.biorxiv.org/content/10.1101/418939v2). For an additional example of MIND in action, check out this excellent work by [Nieh et al. 2021](https://www.nature.com/articles/s41586-021-03652-7).  

The schematic below presents the internal workings of MIND (adapted from [Low et al. 2018](https://www.biorxiv.org/content/10.1101/418939v2)):
><img src="https://github.com/qferryAI/annotated_MIND/blob/main/pics/mind_schematic.png" style='width: 75%'>

The algorithm analyzes a series of trajectories (i.e., sequence of states) in a high-dimensional state space (here neural activity) and attempt to extract the low-dimensional manifold they are believed to be restricted to. In order to do that, MIND implements the following steps:
1. Fit a random forest to the data. The forest is composed of N trees that partition the states into local neighborhoods and fit a generative model ([probabilistic pca](https://www.robots.ox.ac.uk/~cvrg/hilary2006/ppca.pdf)) over successor states for each neighborhood. 
2. Use inferred transision probabilities from the random forest and a shortest path algorithm to computes global distances between state pairs in the state space.
3. Embed data in a lower dimensional space using [multidimensional scaling](https://en.wikipedia.org/wiki/Multidimensional_scaling) on the global distance matrix, followed with optional gradient-based optimization of the coordinates.
4. Find optimal [locally linear embedding](https://cs.nyu.edu/~roweis/lle/papers/lleintro.pdf) (LLE) parameters to map novel states to the manifold and back.

## About this code

This implementation follows closely the method of the original paper:
> [Low, R. J., Lewallen, S., Aronov, D., Nevers, R. & Tank, D. W. 
Probing variability in a cognitive map using manifold inference from neural dynamics. 
Biorxiv 418939 (2018) doi:10.1101/418939.](https://www.biorxiv.org/content/10.1101/418939v2)

All functions necesseray to generate MIND embeddings, as well as perform forward and reverse transforms between high and low-dimensional spaces, are conveniently wrapped up as methods of the `MIND` class (see [mind.py](https://github.com/qferryAI/annotated_MIND/blob/main/mind.py) for annotated code). The `MIND` class follows the scikit-learn syntax and implements the following methods:
```python
mind = MIND(*args, **kwargs) # creates a MIND instance
mind.fit(X, Xp, *args, **kwargs) # generates MIND embedding from dynamical data
Y = mind.transform(X, *args, **kwargs) # maps high-dimensional state to low-dimensional embedding
X = mind.inverse_transform(Y, *args, **kwargs) # performs reverse mapping
```
See [this jupyter notebook](https://github.com/qferryAI/annotated_MIND/blob/main/mind_demo.ipynb) for detail explanations and demo on synthetic data. In this notebook, you'll learn to apply MIND to recover a 2-dimensional [Swiss roll manifold](https://scikit-learn.org/stable/auto_examples/manifold/plot_swissroll.html) embedded into a 3-dimensional state spaces as shown below: 
><img src="https://github.com/qferryAI/annotated_MIND/blob/main/pics/mind_notebook_01.png" style='width: 75%'>
Additionally, you will learn to perform forward and reverse mapping between these two spaces:
><img src="https://github.com/qferryAI/annotated_MIND/blob/main/pics/mind_notebook_02.png" style='width: 75%'>

## Module requirements

This code relies on the following Python modules:
- [numpy](https://numpy.org/doc/#) for everything linear algebra.
- [autograd](https://github.com/HIPS/autograd) for gradient-based optimization.
- [scipy](https://scipy.org/) for mutilvariate normal & shortest path algorithm.
- [scikit-learn](https://scikit-learn.org/stable/) for kNN & dimensionality reduction algorithms.
- [matplotlib](https://matplotlib.org/) for visualization.

To make things easier, I am providing the [annotated_MIND.yaml](https://github.com/qferryAI/annotated_MIND/blob/main/annotated_MIND.yaml) file of the conda environment I used for this project. To build and use the same environment, make sure you have [conda](https://docs.conda.io/en/latest/) install on your machine, clone this repo, navigate to the folder, and enter the following command line in your terminal (or [Anaconda prompt](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) for Windows users):
`conda env create -f annotated_MIND.yaml`.

## Enjoy! :wink:




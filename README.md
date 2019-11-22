# COMS 4995-004 Final Project (Fall '19)

##### Shunhua Jiang, Clayton Sanford, Carolina Zheng

---
We present experimental results for a modification of accelerated gradient descent introduced in [Jin et al., 2017](https://arxiv.org/abs/1711.10456).

To reproduce the experiments found in our report, run `simple_functions_test.py` to automatically generate plots for the non-ML functions. For the ML experiments, first run `main.py`, which will save training logs to a `logs` folder. These can be turned into plots using the `plot_ml.ipynb` notebook.

Note that the training datasets ([MNIST](https://pytorch.org/docs/stable/_modules/torchvision/datasets/mnist.html) from torchvision and [MovieLens](https://grouplens.org/datasets/movielens/)) will automatically be downloaded into a `data` folder when the scripts are run.

To run a grid search over the optimizer parameters, run `grid_search.py` and/or `non_ml_grid_search.py`. These scripts already contain the parameter ranges we searched over.

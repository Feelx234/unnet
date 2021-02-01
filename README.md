`unnet` is a library that can be used to study edge-uncertainty in networks for different network analysis tasks. See related preprint https://arxiv.org/abs/2010.11546.

To study edge uncertainty we start from an initial network which can be a real network or synthetic network. You can then apply different types of edge uncertainty, such as Jaccard noise or node-label based noise which systematically remove or add edges. Finally change can be quantified for subsequent network analysis tasks.

In unnet we provide a network generator for the barabasi+ homophily model in `generators.py`. Different samplers can be found in `samplers.py`. As network analysis task we provide wrappers for common network centrality measures in `centralities.py`.

# Example usage

In the notebooks folder we have included several jupyter notebooks which can be executed to reproduce the experiments in a few minutes.
In particular the notebook `submission_plots_BA.ipynb` can be run without the data.
If you plan on using the real world networks they should be put in the notebooks folder as well (They are expected in `.csv` format, you can convert them using the provided `*.gexf` converter.


# Installation instructions
This software uses the graph tool library https://graph-tool.skewed.de/ which is relatively difficult to install on non linux operation systems (see https://git.skewed.de/count0/graph-tool/-/wikis/installation-instructions).
After installing graph tool you can install this library like any other python package, depending packages are also installed along with it.
```
git clone git@github.com:Feelx234/unnet.git
pip install -e unnet
```
The installation should only take a few minutes.

# System requirements
The unnet package should run on any modern standard computer.


## Software requirements
### OS Requirements
This package has been tested on *macOS* and *Linux*:
+ macOS: Mojave (10.14.1)
+ Linux: Ubuntu 20.04.1 LTS

### Python Dependencies
The code was tested with Python 3.8.5.
`unnet` depends on the Python scientific stack.

```
numpy
pandas
matplotlib
tqdm
```
Parts of it also depend on the just-in-time compiler `numba` we used version 0.50.1.

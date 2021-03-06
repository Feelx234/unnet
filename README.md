`unnet` is a library that can be used to study edge-uncertainty in networks for different network analysis tasks. See related preprint https://arxiv.org/abs/2010.11546.

To study edge uncertainty we start from an initial network which can be a real network or synthetic network. You can then apply different types of edge uncertainty, such as Jaccard noise or node-label based noise which systematically remove or add edges. Finally change can be quantified for subsequent network analysis tasks.

In unnet we provide a network generator for the barabasi+ homophily model in `generators.py`. Different samplers can be found in `samplers.py`. As network analysis task we provide wrappers for common network centrality measures in `centralities.py`.

# Example usage

In the notebooks folder we have included several jupyter notebooks which can be executed to reproduce the experiments in a few minutes.
In particular the notebook `submission_plots_BA.ipynb` can be run without the data.
If you plan on using the real world networks, the public available ones can be downloaded by executing 
```
python get_datasets.py
```
which stores them in the notebooks folder. You can then also run the `submission_plots_real.ipynb` notebook.


# Installation instructions
This software uses the graph tool library https://graph-tool.skewed.de/ which is relatively difficult to install on non linux operation systems (see https://git.skewed.de/count0/graph-tool/-/wikis/installation-instructions).
After installing graph tool you can install this library like any other python package, depending packages are also installed along with it.

### Installing conda
```
wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
chmod +rwx Anaconda3-2020.11-Linux-x86_64.sh
bash Anaconda3-2020.11-Linux-x86_64.sh
```
Now follow the installation instructions for anaconda. Afterwards reopen the console to make sure anaconda is active.
### Installing graph tool
```
conda create --name conda_unnet -c conda-forge python=3.8 graph-tool
conda activate conda_unnet
```
### Installing unnet
```
git clone https://github.com/Feelx234/unnet.git
pip install -e unnet
```
The installation should take less than a minute.
If you also want to run the demo notebooks please install `jupyter` as outlined below.
```
pip install jupyter
python -m ipykernel install --user --name=conda_unnet
python -m jupyter notebook
```
Now inside your jupyter make sure you are using the conda_unnet kernel and you are good to go.


# System requirements
The unnet package should run on any modern standard computer.


## Software requirements
### OS Requirements
This package has been tested on *macOS* and *Linux*:
+ macOS: Big Sur (11.1)
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

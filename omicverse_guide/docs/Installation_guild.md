# Installation

## Prerequisites


OmicVerse can be installed via conda or pypi and you need to install `pytorch` at first

!!! note 
    To avoid potential dependency conflicts, installing within a `conda` environment is recommended. And using `pip install -U omicverse` to update.

### Platform

In different platform, there are some differences in the most appropriate installation method.

- `Windows`: We recommend installing the [`wsl` subsystem](https://learn.microsoft.com/en-us/windows/wsl/install) and installing `conda` in the wsl subsystem to configure the omicverse environment.
- `Linux`: We can choose to install [anaconda](https://www.anaconda.com/) or [miniconda](https://docs.conda.io/en/latest/miniconda.html), and then use conda to configure the omicverse environment
- `Mac Os`: We recommend using [`miniforge`](https://github.com/conda-forge/miniforge)  or [`mambaforge`](https://www.rho-signal-effective-analytics.com/modules/pre-course/miniconda-installation/)to configure.

### pip prerequisites
- If using conda/mamba, then just run `conda install -c anaconda pip` and skip this section.
- Install Python, we prefer the pyenv version management system, along with pyenv-virtualenv.

### Apple silicon prerequisites
Installing omicverse on a Mac with Apple Silicon is only possible using a native version of python. A native version of python can be installed with an Apple Silicon version of mambaforge (which can be installed from a native version of homebrew via `brew install --cask mambaforge`). 

## Conda

1.  Install conda. We typically use the `mambaforge` distribution. Use python>=3.8, conda consider using mamba instead of conda.
2.  Create a new conda environment: 

   ```shell
   conda create -n omicverse python=3.8
   ```
3.  Activate your environment:

   ```shell
   conda activate omicverse
   ```
4.  Install [PyTorch](https://pytorch.org/get-started/locally/) at first:

   ```shell
   conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
   ```
5.  Install `omicverse`:

   ```shell
   conda install omicverse -c conda-forge
   ```

## Pip

The `omicverse` package can be installed via pip using one of the following commands:

1. Install [PyTorch](https://pytorch.org/get-started/locally/) at first: More about the installation can be found at [PyTorch](https://pytorch.org/get-started/locally/). 

   ```shell
   # ROCM 5.2 (Linux only)
   pip3 install torch torchvision torchaudio --extra-index-url
   pip install torch==1.13.1+rocm5.2 torchvision==0.14.1+rocm5.2 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/rocm5.2
   # CUDA 11.6
   pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
   # CUDA 11.7
   pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
   # CPU only
   pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu
   ```
2. After the installation of pytorch, we can start to install `omicverse` by `pip`

   ```shell
   pip install -U omicverse
   pip install -U numba
   ```
3. If you want to using Nightly verseion. There are two ways for you to install

   - Nightly version - clone this [repo](https://github.com/Starlitnightly/omicverse) and run: `pip install .`
   - Using `pip install git+https://github.com/Starlitnightly/omicverse.git`



## Development

For development - clone this repo and run:

```shell
pip install -e ".[dev,docs]"
```


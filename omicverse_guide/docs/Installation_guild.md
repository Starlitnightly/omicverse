# Installation

## Prerequisites

OmicVerse now only can be installed by pypi and you need to install `pytorch` at first

!!! note 
    To avoid potential dependency conflicts, installing within a pip environment is recommended.

### pip prerequisites
- If using conda/mamba, then just run `conda install -c anaconda pip` and skip this section.
- Install Python, we prefer the pyenv version management system, along with pyenv-virtualenv.

### Apple silicon prerequisites
Installing omicverse on a Mac with Apple Silicon is only possible using a native version of python. A native version of python can be installed with an Apple Silicon version of mambaforge (which can be installed from a native version of homebrew via `brew install --cask mambaforge`). 

## Pip

The `omicverse` package can be installed via pip using one of the following commands:

### Pytorch

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

More about the installation can be found at [PyTorch](https://pytorch.org/get-started/locally/). After the installation of pytorch, we can start to install `omicverse` by `pip`


```shell
pip install -U omicverse
pip install -U numba
```

Nightly version - clone this repo and run:

```shell
pip install .
```

## Development

For development - clone this repo and run:

```shell
pip install -e ".[dev,docs]"
```


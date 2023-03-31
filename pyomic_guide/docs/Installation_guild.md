# Installation

## Prerequisites

Pyomic now only can be installed by pypi

!!! note 
    To avoid potential dependency conflicts, installing within a pip environment is recommended.

### pip prerequisites
- If using conda/mamba, then just run conda install -c anaconda pip and skip this section.
- Install Python, we prefer the pyenv version management system, along with pyenv-virtualenv.

### Apple silicon prerequisites
Installing Pyomic on a Mac with Apple Silicon is only possible using a native version of python. A native version of python can be installed with an Apple Silicon version of mambaforge (which can be installed from a native version of homebrew via `brew install --cask mambaforge`). 

## Pip

The `Pyomic` package can be installed via pip using one of the following commands:

```
pip install -U Pyomic
```
Nightly version - clone this repo and run:
```
pip install .
```

## Development
For development - clone this repo and run:
```
pip install -e ".[dev,docs]"
```



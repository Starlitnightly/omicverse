# OmicVerse Installation Guide

For the Chinese version, please check [安装指南 (中文版)](Installation_guide_zh.md).

## Prerequisites

OmicVerse can be installed via conda or pip, but you must install **PyTorch** first.

:::{note}
We recommend installing OmicVerse within a `conda` environment to avoid dependency
conflicts. Use `pip install -U omicverse` to update existing installations.

We also recommend using `uv pip` instead of regular `pip`.
You can install `uv` by running `pip install uv`.
:::

### Platform-Specific Requirements

:::::{tab-set}

::::{tab-item} Windows (WSL)
Install the [WSL subsystem](https://learn.microsoft.com/en-us/windows/wsl/install)
and configure conda within WSL.
::::

::::{tab-item} Windows (Native)
Starting from version `1.6.2`, OmicVerse supports native Windows.
You'll need to install `torch` and `torch_geometric` first.
::::

::::{tab-item} Linux
Install [Anaconda](https://www.anaconda.com/) or
[Miniconda](https://docs.conda.io/en/latest/miniconda.html).
::::

::::{tab-item} macOS
Use [`miniforge`](https://github.com/conda-forge/miniforge) or
[`mambaforge`](https://www.rho-signal-effective-analytics.com/modules/pre-course/miniconda-installation/).

**Important for Apple Silicon Macs:** OmicVerse requires a native version of Python.
Install a native Apple Silicon version of mambaforge using Homebrew:

```shell
brew install --cask mambaforge
```
::::

:::::

## Installation Methods

:::::{tab-set}

::::{tab-item} Quick Install (Recommended)
:sync: quick

The easiest way to install OmicVerse is using our installation script:

```shell
# Linux only
curl -sSL omicverse.com/install | bash -s
```

This script will automatically:

- Set up the appropriate environment
- Install the correct PyTorch version for your system
- Install all required dependencies
- Configure OmicVerse optimally for your hardware
::::

::::{tab-item} Conda / Mamba
:sync: conda

1. **Create and activate a new environment**:

   ```shell
   conda create -n omicverse python=3.10
   conda activate omicverse
   ```

2. **Install PyTorch and PyTorch Geometric (PyG)**:

   ```shell
   # For CUDA support (check your CUDA version with 'nvcc --version')
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

   # OR for CPU-only installation
   conda install pytorch torchvision torchaudio cpuonly -c pytorch

   # Install PyTorch Geometric
   conda install pyg -c pyg
   ```

3. **Install OmicVerse**:

   ```shell
   conda install omicverse -c conda-forge
   ```

4. **Verify the installation**:

   ```shell
   python -c "import omicverse"
   ```
::::

::::{tab-item} pip / PyPI
:sync: pip

1. **Install uv (recommended package manager)**:

   ```shell
   pip install uv
   ```

2. **Install PyTorch** *(installing with pip on macOS may encounter some issues)*:

   ```shell
   uv pip install torch torchvision torchaudio
   ```

3. **Install PyTorch Geometric**:

   ```shell
   uv pip install torch_geometric
   ```

4. **Install OmicVerse**:

   ```shell
   uv pip install omicverse
   ```

5. **Verify the installation**:

   ```shell
   python -c "import omicverse"
   ```
::::

:::::

## Other Options

:::::{tab-set}

::::{tab-item} Nightly / Development Build

To install the latest development version with the newest features:

```shell
# Option 1: Clone repository and install locally
git clone https://github.com/Starlitnightly/omicverse.git
cd omicverse
pip install .

# Option 2: Install directly from GitHub
pip install git+https://github.com/Starlitnightly/omicverse.git
```
::::

::::{tab-item} Developer Setup

For contributors:

```shell
pip install -e ".[dev,docs]"
```
::::

::::{tab-item} GPU-Accelerated (RAPIDS)

For maximum performance with GPU acceleration:

```shell
# 1. Create a new conda environment
conda create -n rapids python=3.11

# 2. Install RAPIDS
conda install rapids=24.04 -c rapidsai -c conda-forge -c nvidia -y

# 3. Install additional RAPIDS components
conda install cudf=24.04 cuml=24.04 cugraph=24.04 cuxfilter=24.04 \
    cucim=24.04 pylibraft=24.04 raft-dask=24.04 cuvs=24.04 \
    -c rapidsai -c conda-forge -c nvidia -y

# 4. Install rapids-singlecell
pip install rapids-singlecell

# 5. Install OmicVerse
curl -sSL https://raw.githubusercontent.com/Starlitnightly/omicverse/refs/heads/master/install.sh | bash -s
```

:::{note}
We install RAPIDS 24.04 because some systems have glibc < 2.28.
Follow the [official RAPIDS tutorial](https://docs.rapids.ai/install) to install
the latest version if your system supports it.
:::
::::

:::::

## Docker

Pre-built Docker images are available on
[Docker Hub](https://hub.docker.com/r/starlitnightly/omicverse).

```shell
docker pull starlitnightly/omicverse
```

## Jupyter Lab Setup

We recommend using Jupyter Lab for interactive analysis:

```shell
pip install jupyterlab
```

After installation, activate your omicverse environment and run `jupyter lab` in your
terminal. A URL will appear that you can open in your browser.

<img src="img/light_jupyter.jpg" class="only-light" alt="Jupyter Lab (light mode)" style="max-width:100%;" />
<img src="img/dark_jupyter.jpg" class="only-dark" alt="Jupyter Lab (dark mode)" style="max-width:100%;" />

## Troubleshooting

:::::{tab-set}

::::{tab-item} Linux GCC

```shell
# Ubuntu
sudo apt update
sudo apt install build-essential

# CentOS
sudo yum group install "Development Tools"

# Verify GCC
gcc --version
```
::::

::::{tab-item} Package Installation Issues

If pip fails to install certain packages (e.g., `scikit-misc`), try conda instead:

```shell
conda install scikit-misc -c conda-forge -c bioconda
```
::::

::::{tab-item} Apple Silicon (M1/M2)

```shell
conda install s_gd2 -c conda-forge
pip install -U omicverse
conda install pytorch::pytorch torchvision torchaudio -c pytorch
```

**Important:** OmicVerse requires a native version of Python on Apple Silicon Macs.
Install a native Apple Silicon version of mambaforge using Homebrew:

```shell
brew install --cask mambaforge
```
::::

::::{tab-item} macOS `omp_set_nested` Deprecated

```shell
# 1. Uninstall pip wheels
pip uninstall -y numpy scipy scikit-learn threadpoolctl \
    torch torchvision torchaudio pytorch-lightning

# 2. Install clean LP64 + OpenBLAS stack from conda-forge
mamba install -c conda-forge \
    "numpy>=1.26,<2" "scipy>=1.11,<2" anndata "scanpy>=1.10" pandas \
    scikit-learn numexpr threadpoolctl \
    "libblas=*=*openblas" "libopenblas=*=*openmp" libomp

# 3. Install PyTorch with conda
mamba install -c pytorch -c conda-forge pytorch torchvision torchaudio
```
::::

:::::

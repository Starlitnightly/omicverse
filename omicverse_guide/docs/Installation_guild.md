# 🧬 OmicVerse Installation Guide

> 📚 For the Chinese version, please check [安装指南 (中文版)](Installation_guide_zh.md)

## 📋 Prerequisites

OmicVerse can be installed via conda or pip, but you must install `PyTorch` first.

!!! note 

    We recommend installing OmicVerse within a `conda` environment to avoid dependency conflicts. Use `pip install -U omicverse` to update existing installations.

    We also recommend using `uv pip` instead of regular `pip`. You can install `uv` by running `pip install uv`.

### Platform-Specific Requirements

=== "Windows (WSL)"

    Install the [WSL subsystem](https://learn.microsoft.com/en-us/windows/wsl/install) and configure conda within WSL.

=== "Windows (Native)"

    Starting from version `1.6.2`, OmicVerse supports native Windows. You'll need to install `torch` and `torch_geometric` first.

=== "Linux"

    Install [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
    
=== "macOS"

    Use [`miniforge`](https://github.com/conda-forge/miniforge) or [`mambaforge`](https://www.rho-signal-effective-analytics.com/modules/pre-course/miniconda-installation/).
    
    **Important for Apple Silicon Macs:** OmicVerse requires a native version of Python. Install a native Apple Silicon version of mambaforge using Homebrew: `brew install --cask mambaforge`.


## 🚀 Installation Methods

=== "Quick Installation (Recommended)"

    !!! note "Quick Installation"

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

=== "Conda/Mamba"

    !!! note "Conda/Mamba"

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

=== "pip/Pypi"

    !!! note "pip/Pypi"

        1. **Install uv (recommended package manager)**:
            ```shell
            pip install uv
            ```
        2. **Install PyTorch**:
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

## Other Important Options


!!! tip "Nightly Version (Latest Development Build)"

    To install the latest development version with newest features:

    ```shell
    # Option 1: Clone repository and install locally
    git clone https://github.com/Starlitnightly/omicverse.git
    cd omicverse
    pip install .

    # Option 2: Install directly from GitHub
    pip install git+https://github.com/Starlitnightly/omicverse.git
    ```

!!! tip "Development Setup"

    For developers who want to contribute to OmicVerse:

    ```shell
    pip install -e ".[dev,docs]"
    ```

!!! tip "GPU-Accelerated Installation (with RAPIDS)"

    For maximum performance with GPU acceleration:

    ```shell
    # 1. Create a new conda environment
    conda create -n rapids python=3.11
    
    # 2. Install RAPIDS using conda
    conda install rapids=24.04 -c rapidsai -c conda-forge -c nvidia -y   
    
    # 3. Install additional RAPIDS components
    conda install cudf=24.04 cuml=24.04 cugraph=24.04 cuxfilter=24.04 cucim=24.04 pylibraft=24.04 raft-dask=24.04 cuvs=24.04 -c rapidsai -c conda-forge -c nvidia -y   
    
    # 4. Install rapids-singlecell
    pip install rapids-singlecell
    
    # 5. Install OmicVerse
    curl -sSL https://raw.githubusercontent.com/Starlitnightly/omicverse/refs/heads/master/install.sh | bash -s
    ```
    
    **Note:** We install RAPIDS version 24.04 because some systems have glibc<2.28. You can follow the official RAPIDS tutorial to install the latest version if your system supports it.

## Docker Installation

Pre-built Docker images are available on [Docker Hub](https://hub.docker.com/r/starlitnightly/omicverse).

## Jupyter Lab Setup

!!! note "Jupyter Lab"

    We recommend using Jupyter Lab for interactive analysis:

    ```shell
    pip install jupyter-lab
    ```

    After installation, activate your omicverse environment and run `jupyter-lab` in your terminal. A URL will appear that you can open in your browser.


![jupyter-light](img/light_jupyter.jpg#gh-light-mode-only)
![jupyter-dark](img/dark_jupyter.jpg#gh-dark-mode-only)



## Troubleshooting

!!! info "Linux GCC Setup"

    ```shell
    # Ubuntu
    sudo apt update
    sudo apt install build-essential

    # CentOS
    sudo yum group install "Development Tools"

    # Verify GCC
    gcc --version
    ```

!!! info "Package Installation Issues"

    If pip fails to install certain packages (e.g., scikit-misc), try using conda instead:
    ```shell
    conda install scikit-misc -c conda-forge -c bioconda
    ```

!!! info "Apple Silicon (M1/M2) Issues"

    For Apple Silicon Mac users experiencing issues:
    ```shell
    conda install s_gd2 -c conda-forge
    pip install -U omicverse
    conda install pytorch::pytorch torchvision torchaudio -c pytorch
    ```

!!! info "Apple Silicon Requirements"

    **Important:** OmicVerse requires a native version of Python on Apple Silicon Macs. Make sure to install a native Apple Silicon version of mambaforge using Homebrew:
    ```shell
    brew install --cask mambaforge
    ```

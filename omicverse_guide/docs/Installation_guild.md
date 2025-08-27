# 🧬 OmicVerse Installation Guide

> 📚 For Chinese version, please check [安装指南 (中文版)](Installation_guide_zh.md)

## 📋 Prerequisites

OmicVerse can be installed via conda or pip, but you must install `PyTorch` first.

!!! note 

    We recommend installing within a `conda` environment to avoid dependency conflicts. Use `pip install -U omicverse` to update existing installations.

    We also recommend using within `uv pip` instead of `pip`, you can run `pip install uv` to install `uv`.

### Platform-Specific Requirements

=== "Windows (WSL)"

    Install the [WSL subsystem](https://learn.microsoft.com/en-us/windows/wsl/install) and configure conda within WSL

=== "Windows (Native)"

    From version `1.6.2`, OmicVerse supports native Windows (requires `torch` and `torch_geometric`)

=== "Linux"

    Install [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
    
=== "macOS"

    Use [`miniforge`](https://github.com/conda-forge/miniforge) or [`mambaforge`](https://www.rho-signal-effective-analytics.com/modules/pre-course/miniconda-installation/)
    OmicVerse requires a native version of Python on Apple Silicon Macs. Install using a native Apple Silicon version of mambaforge (available via Homebrew with `brew install --cask mambaforge`).


## 🚀 Installation Methods

=== "Quick Installation (Recommended)"

    !!! note "Quick Installation"

        The easiest way to install OmicVerse is using our installation script:

        ```shell
        #Only for Linux
        curl -sSL https://raw.githubusercontent.com/Starlitnightly/omicverse/refs/heads/master/install.sh | bash -s
        ```

        This script will automatically:
        - Set up the appropriate environment
        - Install the correct PyTorch version for your system
        - Install all required dependencies
        - Configure OmicVerse optimally for your hardware

=== "Conda/Mamba"

    !!! note "Conda/Mamba"

        1. **Create and activate environment**:
          ```shell
          conda create -n omicverse python=3.10
          conda activate omicverse
          ```

        2. **Install PyTorch and PyG**:
          ```shell
          # For CUDA (check your version with 'nvcc --version')
          conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
          
          # OR for CPU only
          conda install pytorch torchvision torchaudio cpuonly -c pytorch
          
          # Install PyG
          conda install pyg -c pyg
          ```

        3. **Install OmicVerse**:
          ```shell
          conda install omicverse -c conda-forge
          ```

        4. **Verify installation**:
          ```shell
          python -c "import omicverse"
          ```

=== "pip/Pypi"

    !!! note "pip/Pypi"

        1. **Install uv**
            ```shell
            pip install uv
            ```
        2. **Install torch**
            ```shell
            uv pip install torch torchvision torchaudio
            ```
        3. **Install PyG Extensions**
            ```shell
            uv pip install torch_geometric
            ```
        4. **Install OmicVerse**:
          ```shell
          uv pip install omicverse
          ```
        5. **Verify installation**:
          ```shell
          python -c "import omicverse"
          ```

## Other Importantance


!!! tip "Nightly Version"

    ```shell
    # Option 1: Clone and install
    git clone https://github.com/Starlitnightly/omicverse.git
    cd omicverse
    pip install .

    # Option 2: Direct install from GitHub
    pip install git+https://github.com/Starlitnightly/omicverse.git
    ```

!!! tip "Development Setup"

    For development:

    ```shell
    pip install -e ".[dev,docs]"
    ```

!!! tip "GPU-Accelerated Installation"

    ```shell
    #1. create a new conda env
    conda create -n rapids python=3.11
    #2. install rapids using conda
    conda install rapids=24.04 -c rapidsai -c conda-forge -c nvidia -y   
    #3. install cuml
    conda install cudf=24.04 cuml=24.04 cugraph=24.04 cuxfilter=24.04 cucim=24.04 pylibraft=24.04 raft-dask=24.04 cuvs=24.04 -c rapidsai -c conda-forge -c nvidia -y   
    #4. install rapid_single_cell
    pip install rapids-singlecell
    #5. install omicverse
    curl -sSL https://raw.githubusercontent.com/Starlitnightly/omicverse/refs/heads/master/install.sh | bash -s
    ```
    Here, we install the rapids==24.04, that's because our system's glibc<2.28. You can follow the official tutorial to install the latest version of rapids.

## Docker 

Docker images are available on [Docker Hub](https://hub.docker.com/r/starlitnightly/omicverse).

## Jupyter Lab Setup

!!! note "Jupyer Lab"

    We recommend Jupyter Lab for interactive analysis:

    ```shell
    pip install jupyter-lab
    ```

    After installation, run `jupyter-lab` in your terminal (from the omicverse environment). A URL will appear that you can open in your browser.


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

!!! info "Package installation issues"

    If pip cannot install certain packages (e.g., scikit-misc), try conda:
    ```shell
    conda install scikit-misc -c conda-forge -c bioconda
    ```
!!! info "Apple Silicon (M1/M2) issues"

    ```shell
    conda install s_gd2 -c conda-forge
    pip install -U omicverse
    conda install pytorch::pytorch torchvision torchaudio -c pytorch
    ```


!!! info "Apple Silicon Note"

    OmicVerse requires a native version of Python on Apple Silicon Macs. Install using a native Apple Silicon version of mambaforge (available via Homebrew with `brew install --cask mambaforge`).

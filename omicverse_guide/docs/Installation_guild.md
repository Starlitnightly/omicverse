# 🧬 OmicVerse Installation Guide

> 📚 For Chinese version, please check [安装指南 (中文版)](Installation_guide_zh.md)

## 📋 Prerequisites

OmicVerse can be installed via conda or pip, but you must install `PyTorch` first.

!!! note 
    We recommend installing within a `conda` environment to avoid dependency conflicts. Use `pip install -U omicverse` to update existing installations.

### Platform-Specific Requirements

- **Windows (WSL)**: Install the [WSL subsystem](https://learn.microsoft.com/en-us/windows/wsl/install) and configure conda within WSL
- **Windows (Native)**: From version `1.6.2`, OmicVerse supports native Windows (requires `torch` and `torch_geometric`)
- **Linux**: Install [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- **macOS**: Use [`miniforge`](https://github.com/conda-forge/miniforge) or [`mambaforge`](https://www.rho-signal-effective-analytics.com/modules/pre-course/miniconda-installation/)

### pip Prerequisites
- If using conda/mamba: Run `conda install -c anaconda pip` and skip this section
- Otherwise: Install Python (preferably using pyenv with pyenv-virtualenv)

### Apple Silicon Note
OmicVerse requires a native version of Python on Apple Silicon Macs. Install using a native Apple Silicon version of mambaforge (available via Homebrew with `brew install --cask mambaforge`).

## 🚀 Installation Methods

### 🔥 Quick Installation (Recommended)

The easiest way to install OmicVerse is using our installation script:

```shell
#Only for Linux
pip3 install torch torchvision torchaudio
curl -sSL https://raw.githubusercontent.com/Starlitnightly/omicverse/refs/heads/master/install.sh | bash -s
```

This script will automatically:
- Set up the appropriate environment
- Install the correct PyTorch version for your system
- Install all required dependencies
- Configure OmicVerse optimally for your hardware

### 📦 Using Conda

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

### 📦 Using pip

<ol>
<li><strong>Install PyTorch</strong>:
   <pre><code class="language-bash"># For CUDA 11.8
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
# OR for CPU only
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu</code></pre>
</li>

<li><strong>Install PyG</strong>:
   <pre><code class="language-bash"># Install base PyG
pip install torch_geometric
   
# Check versions
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.version.cuda)"</code></pre>
</li>

<li><strong>Install PyG Extensions</strong>: 

   <h4>⚠️ Not Recommended Method</h4>
   <pre><code class="language-bash"># For Windows with CPU
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-${TORCH}+cpu.html
   
# For systems with CUDA
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html</code></pre>

   <p>Replace <code class="language-bash">${TORCH}</code> and <code>${CUDA}</code> with your version numbers:</p>
   
   <table>
     <thead>
       <tr>
         <th>PyTorch Version</th>
         <th>TORCH Value</th>
         <th>CUDA Options</th>
       </tr>
     </thead>
     <tbody>
       <tr>
         <td>PyTorch 2.7</td>
         <td>2.7.0</td>
         <td>cpu/cu122/cu124</td>
       </tr>
       <tr>
         <td>PyTorch 2.6</td>
         <td>2.6.0</td>
         <td>cpu/cu122/cu124</td>
       </tr>
       <tr>
         <td>PyTorch 2.5</td>
         <td>2.5.0</td>
         <td>cpu/cu118/cu121/cu122</td>
       </tr>
       <tr>
         <td>PyTorch 2.4</td>
         <td>2.4.0</td>
         <td>cpu/cu118/cu121</td>
       </tr>
       <tr>
         <td>PyTorch 2.3</td>
         <td>2.3.0</td>
         <td>cpu/cu118/cu121</td>
       </tr>
       <tr>
         <td>PyTorch 2.2</td>
         <td>2.2.0</td>
         <td>cpu/cu118/cu121</td>
       </tr>
       <tr>
         <td>PyTorch 2.1</td>
         <td>2.1.0</td>
         <td>cpu/cu118/cu121</td>
       </tr>
       <tr>
         <td>PyTorch 2.0</td>
         <td>2.0.0</td>
         <td>cpu/cu117/cu118</td>
       </tr>
       <tr>
         <td>PyTorch 1.13</td>
         <td>1.13.0</td>
         <td>cpu/cu116/cu117</td>
       </tr>
     </tbody>
   </table>
   
   <p>Example commands:</p>
   <pre><code class="language-bash"># For PyTorch 2.7 with CUDA 12.4
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.7.0+cu124.html
   
# For PyTorch 2.3 with CUDA 12.1
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
   
# For PyTorch 2.2 with CUDA 11.8
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu118.html</code></pre>

   <h4>✅ Recommended Method</h4>
   <pre><code class="language-bash">conda install -c conda-forge pytorch_scatter pytorch_sparse pytorch_cluster pytorch_spline_conv</code></pre>
</li>

<li><strong>Linux GCC Setup</strong> (Linux only):
   <pre><code class="language-bash"># Ubuntu
sudo apt update
sudo apt install build-essential
   
# CentOS
sudo yum group install "Development Tools"
   
# Verify GCC
gcc --version</code></pre>
</li>

<li><strong>Install OmicVerse</strong>:
   <pre><code class="language-bash"># Basic installation
pip install -U omicverse
   
# Install Numba for performance optimization
pip install -U numba
   
# OR full installation with spatial RNA-seq support
pip install omicverse[full]</code></pre>
</li>

<li><strong>Verify installation</strong>:
   <pre><code class="language-bash">python -c "import omicverse"</code></pre>
</li>
</ol>

## 🔧 Advanced Options

### Nightly Version

```shell
# Option 1: Clone and install
git clone https://github.com/Starlitnightly/omicverse.git
cd omicverse
pip install .

# Option 2: Direct install from GitHub
pip install git+https://github.com/Starlitnightly/omicverse.git
```

### GPU-Accelerated Installation

```shell
# Using conda/mamba
conda env create -f conda/omicverse_gpu.yml
# OR
mamba env create -f conda/omicverse_gpu.yml
```

### Docker

Docker images are available on [Docker Hub](https://hub.docker.com/r/starlitnightly/omicverse).

## 📊 Jupyter Lab Setup

We recommend Jupyter Lab for interactive analysis:

```shell
pip install jupyter-lab
```

After installation, run `jupyter-lab` in your terminal (from the omicverse environment). A URL will appear that you can open in your browser.

![jupyter-light](img/light_jupyter.jpg#gh-light-mode-only)
![jupyter-dark](img/dark_jupyter.jpg#gh-dark-mode-only)

## 🛠️ Development Setup

For development:

```shell
pip install -e ".[dev,docs]"
```

## ❓ Troubleshooting

- **Package installation issues**: If pip cannot install certain packages (e.g., scikit-misc), try conda:
  ```shell
  conda install scikit-misc -c conda-forge -c bioconda
  ```

- **Apple Silicon (M1/M2) issues**:
  ```shell
  conda install s_gd2 -c conda-forge
  pip install -U omicverse
  conda install pytorch::pytorch torchvision torchaudio -c pytorch
  ```

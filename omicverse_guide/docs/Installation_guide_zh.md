# 🧬 OmicVerse 安装指南

> 📚 For English version, please check [Installation Guide (English)](Installation_guild.md)

## 📋 前提条件

OmicVerse 可以通过 conda 或 pip 安装，但首先需要安装 `PyTorch`。

!!! note 
    我们建议在 `conda` 环境中安装，以避免依赖冲突。使用 `pip install -U omicverse` 更新现有安装。

### 平台特定要求

- **Windows (WSL)**: 安装 [WSL 子系统](https://learn.microsoft.com/en-us/windows/wsl/install) 并在 WSL 中配置 conda
- **Windows (原生)**: 从版本 `1.6.2` 开始，OmicVerse 支持原生 Windows（需要先安装 `torch` 和 `torch_geometric`）
- **Linux**: 安装 [Anaconda](https://www.anaconda.com/) 或 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- **macOS**: 使用 [`miniforge`](https://github.com/conda-forge/miniforge) 或 [`mambaforge`](https://www.rho-signal-effective-analytics.com/modules/pre-course/miniconda-installation/)

### pip 前提条件
- 如果使用 conda/mamba: 运行 `conda install -c anaconda pip` 并跳过此部分
- 否则: 安装 Python（最好使用 pyenv 和 pyenv-virtualenv 进行版本管理）

### Apple Silicon 注意事项
在搭载 Apple Silicon 的 Mac 上，OmicVerse 只能使用原生版本的 Python 安装。您可以通过原生版本的 Homebrew 安装原生版本的 mambaforge（使用 `brew install --cask mambaforge`）。

## 🚀 安装方法

### 🔥 快速安装（推荐）

安装 OmicVerse 最简单的方法是使用我们的安装脚本：

```shell
#仅适用于Linux
curl -sSL https://raw.githubusercontent.com/Starlitnightly/omicverse/refs/heads/master/install.sh | bash -s
```

该脚本会自动：
- 设置适当的环境
- 为您的系统安装正确的 PyTorch 版本
- 安装所有必需的依赖项
- 为您的硬件优化配置 OmicVerse

### 📦 使用 Conda

1. **创建并激活环境**:
   ```shell
   conda create -n omicverse python=3.10
   conda activate omicverse
   ```

2. **安装 PyTorch 和 PyG**:
   ```shell
   # 对于 CUDA（使用 'nvcc --version' 检查您的版本）
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   
   # 或仅使用 CPU
   conda install pytorch torchvision torchaudio cpuonly -c pytorch
   
   # 安装 PyG
   conda install pyg -c pyg
   ```

3. **安装 OmicVerse**:
   ```shell
   conda install omicverse -c conda-forge
   ```

4. **验证安装**:
   ```shell
   python -c "import omicverse"
   ```

### 📦 使用 pip

<ol>
<li><strong>安装 PyTorch</strong>:
   <pre><code class="language-bash"># 对于 CUDA 11.8
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
# 或仅使用 CPU
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu</code></pre>
</li>

<li><strong>安装 PyG</strong>:
   <pre><code class="language-bash"># 安装基础 PyG
pip install torch_geometric
   
# 检查版本
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.version.cuda)"</code></pre>
</li>

<li><strong>安装 PyG 扩展</strong>: 

   <h4>⚠️ 不推荐方法</h4>
   <pre><code class="language-bash"># 对于仅使用 CPU 的 Windows
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-${TORCH}+cpu.html
   
# 对于使用 CUDA 的系统
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html</code></pre>

   <p>将 <code>${TORCH}</code> 和 <code>${CUDA}</code> 替换为您的版本号：</p>
   
   <table>
     <thead>
       <tr>
         <th>PyTorch 版本</th>
         <th>TORCH 值</th>
         <th>CUDA 选项</th>
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
   
   <p>示例命令：</p>
   <pre><code class="language-bash"># 对于 PyTorch 2.7 和 CUDA 12.4
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.7.0+cu124.html
   
# 对于 PyTorch 2.3 和 CUDA 12.1
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
   
# 对于 PyTorch 2.2 和 CUDA 11.8
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu118.html</code></pre>

   <h4>✅ 推荐方法</h4>
   <pre><code class="language-bash">conda install -c conda-forge pytorch_scatter pytorch_sparse pytorch_cluster pytorch_spline_conv</code></pre>
</li>

<li><strong>Linux GCC 配置</strong>（仅适用于 Linux）:
   <pre><code class="language-bash"># Ubuntu
sudo apt update
sudo apt install build-essential
   
# CentOS
sudo yum group install "Development Tools"
   
# 验证 GCC
gcc --version</code></pre>
</li>

<li><strong>安装 OmicVerse</strong>:
   <pre><code class="language-bash"># 基本安装
pip install -U omicverse
   
# 安装 Numba 以优化性能
pip install -U numba
   
# 或带有空间转录组支持的完整安装
pip install omicverse[full]</code></pre>
</li>

<li><strong>验证安装</strong>:
   <pre><code class="language-bash">python -c "import omicverse"</code></pre>
</li>
</ol>

## 🔧 高级选项

### 开发版本

```shell
# 选项 1: 克隆并安装
git clone https://github.com/Starlitnightly/omicverse.git
cd omicverse
pip install .

# 选项 2: 直接从 GitHub 安装
pip install git+https://github.com/Starlitnightly/omicverse.git
```

### GPU 加速安装

```shell
# 使用 conda/mamba
conda env create -f conda/omicverse_gpu.yml
# 或
mamba env create -f conda/omicverse_gpu.yml
```

### Docker

Docker 镜像可在 [Docker Hub](https://hub.docker.com/r/starlitnightly/omicverse) 上获取。

## 📊 Jupyter Lab 设置

我们推荐使用 Jupyter Lab 进行交互式分析：

```shell
pip install jupyter-lab
```

安装完成后，在终端中（从 omicverse 环境下）运行 `jupyter-lab`。将会出现一个 URL，您可以在浏览器中打开它。

![jupyter-light](img/light_jupyter.jpg#gh-light-mode-only)
![jupyter-dark](img/dark_jupyter.jpg#gh-dark-mode-only)

## 🛠️ 开发环境设置

对于开发：

```shell
pip install -e ".[dev,docs]"
```

## ❓ 故障排除

- **包安装问题**: 如果 pip 无法安装某些包（例如 scikit-misc），请尝试使用 conda：
  ```shell
  conda install scikit-misc -c conda-forge -c bioconda
  ```

- **Apple Silicon (M1/M2) 问题**:
  ```shell
  conda install s_gd2 -c conda-forge
  pip install -U omicverse
  conda install pytorch::pytorch torchvision torchaudio -c pytorch
  ```
  
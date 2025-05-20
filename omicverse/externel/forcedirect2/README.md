# ForceAtlas2 Python 实现

这是一个高性能的ForceAtlas2图形布局算法Python实现，支持多线程和GPU加速。

## 性能优化

本实现支持三种运行模式，按性能从高到低排序:

1. **Cython编译 + GPU加速**: 最高性能 (GPU加速 + C级别速度)
2. **Cython编译 + 多线程CPU**: 高性能 (C级别速度 + 并行计算)
3. **纯Python + 多线程/GPU**: 中等性能 (Python解释速度 + 并行计算)
4. **纯Python单线程**: 最低性能 (开发/测试用)

## 如何编译获得最佳性能

当使用未编译的模块时，您会看到这个警告:
```
Warning: uncompiled fa2util module. Compile with cython for a 10-100x speed boost.
```

### 简单编译方法

我们提供了一个简单的编译脚本，运行以下命令即可:

```bash
cd path/to/omicverse/externel/forcedirect2
python compile.py
```

### 手动编译方法

1. 确保已安装Cython和numpy:
```bash
pip install cython numpy
```

2. 在forcedirect2目录下运行:
```bash
python setup.py build_ext --inplace
```

### 编译要求

- **Windows**: 需要安装Visual C++ Build Tools
- **Mac**: 需要安装XCode命令行工具
- **Linux**: 需要安装gcc和python-dev包

## 使用方法

编译完成后，ForceAtlas2算法会自动使用编译后的模块，性能将提升10-100倍。您不需要修改任何代码，只需确保编译产生的.so/.pyd文件位于Python可以找到的路径中。 
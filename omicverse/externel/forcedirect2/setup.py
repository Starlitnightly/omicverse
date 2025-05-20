from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import os
import sys

# 查找是否存在.pyx文件而不是.py文件
force_dir = os.path.dirname(os.path.abspath(__file__))
has_pyx = os.path.exists(os.path.join(force_dir, "fa2util.pyx"))
source_file = "fa2util.pyx" if has_pyx else "fa2util.py"

# 定义编译参数
compiler_directives = {
    'language_level': 3,
    'boundscheck': False,  # 关闭数组边界检查 (提高性能)
    'wraparound': False,   # 关闭负索引自动处理 (提高性能)
    'initializedcheck': False,  # 关闭初始化检查 (提高性能)
    'cdivision': True,     # 使用C除法而不是Python除法 (提高性能)
    'fastcall': True       # 优化函数调用 (提高性能)
}

# 定义额外的编译参数
extra_compile_args = []
if sys.platform == 'win32':
    extra_compile_args.append('/O2')  # Windows: 启用优化
else:
    extra_compile_args.extend(['-O3', '-ffast-math'])  # Unix: 启用高级优化

# 定义扩展模块
extensions = [
    Extension(
        "fa2util",  # 不要包含.pyx后缀
        [source_file],  # Cython源文件
        include_dirs=[numpy.get_include()],  # 添加numpy头文件
        language="c++",  # 使用C++编译
        extra_compile_args=extra_compile_args
    )
]

print(f"正在编译 {source_file}...")
print(f"编译器指令: {compiler_directives}")
print(f"额外编译参数: {extra_compile_args}")

# 安装
setup(
    name="forceatlas2",
    version="1.0.0",
    description="高性能ForceAtlas2算法实现",
    author="OMICVerse Team",
    ext_modules=cythonize(
        extensions,
        compiler_directives=compiler_directives,
        annotate=False  # 设置为True可以生成html注释文件
    ),
    zip_safe=False
) 
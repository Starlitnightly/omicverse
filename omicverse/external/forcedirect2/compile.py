#!/usr/bin/env python
"""
快速编译工具，用于编译fa2util模块以获得10-100倍的性能提升
"""

import os
import sys
import subprocess
import shutil

def clean_build_files():
    """清理之前的编译文件和缓存"""
    print("清理旧的编译文件...")
    
    # 获取当前脚本所在的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 删除编译产生的文件和目录
    extensions = ['.c', '.cpp', '.html', '.so', '.pyd', '.dll']
    files_to_remove = []
    
    # 查找可能的编译输出文件
    for file in os.listdir(script_dir):
        if file.startswith('fa2util') and any(file.endswith(ext) for ext in extensions):
            files_to_remove.append(os.path.join(script_dir, file))
            
    # 删除build目录
    build_dir = os.path.join(script_dir, 'build')
    if os.path.exists(build_dir):
        files_to_remove.append(build_dir)
    
    # 删除__pycache__目录
    pycache_dir = os.path.join(script_dir, '__pycache__')
    if os.path.exists(pycache_dir):
        files_to_remove.append(pycache_dir)
    
    # 执行删除
    for path in files_to_remove:
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
                print(f"已删除目录: {path}")
            else:
                os.remove(path)
                print(f"已删除文件: {path}")
        except Exception as e:
            print(f"无法删除 {path}: {str(e)}")
    
    print("清理完成")

def fix_pxd_file():
    """检查并修复fa2util.pxd文件中的已知问题"""
    pxd_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fa2util.pxd')
    
    if not os.path.exists(pxd_file):
        print(f"警告: 找不到 {pxd_file}")
        return
    
    try:
        with open(pxd_file, 'r') as f:
            content = f.read()
        
        # 修复已知问题: =None 默认值在pxd文件中是不允许的
        if "all_nodes=None" in content:
            print("修复pxd文件中的语法问题...")
            content = content.replace("all_nodes=None", "all_nodes=*")
            with open(pxd_file, 'w') as f:
                f.write(content)
            print("已修复 fa2util.pxd")
    except Exception as e:
        print(f"修复pxd文件时出错: {str(e)}")

def main():
    print("开始编译fa2util模块...")
    
    # 获取当前脚本所在的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 确保我们在正确的目录中
    os.chdir(script_dir)
    
    # 清理旧的编译文件
    clean_build_files()
    
    # 修复pxd文件中的已知问题
    fix_pxd_file()
    
    # 检查是否安装了所需的包
    try:
        import cython
        import numpy
    except ImportError:
        print("缺少必要的依赖，正在安装...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "cython", "numpy"])
    
    # 执行编译
    try:
        subprocess.check_call([sys.executable, "setup.py", "build_ext", "--inplace"])
        
        # 验证编译结果
        compiled_files = [f for f in os.listdir(script_dir) 
                         if f.startswith('fa2util') and (f.endswith('.so') or f.endswith('.pyd'))]
        
        if compiled_files:
            print("编译成功完成！")
            print(f"生成的文件: {', '.join(compiled_files)}")
            print("现在ForceAtlas2应该能以10-100倍的速度运行。")
            return 0
        else:
            print("警告: 编译似乎成功，但找不到编译后的模块文件。")
            print("ForceAtlas2仍将使用较慢的Python实现。")
            return 1
    except subprocess.CalledProcessError as e:
        print(f"编译失败，错误代码: {e.returncode}")
        print("请确保已安装Cython和必要的编译工具。")
        print("对于Windows用户，您可能需要安装Visual C++ Build Tools。")
        print("对于Mac用户，您可能需要安装XCode命令行工具。")
        print("对于Linux用户，请确保已安装gcc和python-dev包。")
        print("\n如果您在服务器上运行，可能需要使用以下命令安装依赖:")
        print("Debian/Ubuntu: sudo apt-get install python3-dev build-essential")
        print("CentOS/RHEL: sudo yum install python3-devel gcc")
        print("\n您也可以尝试使用GPU版本，它不需要编译:")
        print("fle(..., use_gpu=True, compile_cython=False)")
        return 1
    
if __name__ == "__main__":
    sys.exit(main()) 
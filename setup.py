r"""
Shim setup.py
"""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

from setuptools import setup, find_packages  

setup(  
    name = 'Pyomic',  
    version = '1.1.2',
    # keywords = ('chinesename',),  
    description = 'A python framework library for omics analysis',  
    license = 'GNU License',  
    install_requires = ['ERgene','numpy','pandas','matplotlib','sklearn','scipy','networkx','seaborn','datetime','statsmodels','gseapy==0.10.8'],  
    packages = find_packages(),  # 要打包的项目文件夹
    include_package_data=True,   # 自动打包文件夹内所有数据
    author = 'ZehuaZeng',  
    author_email = 'Starlitnightly@163.com',
    url = 'https://github.com/Starlitnightly/Pyomic',
    long_description=long_description,  
    long_description_content_type="text/markdown",  
    python_requires='>=3.6',
    py_modules=['bulk','single'],
    # packages = find_packages(include=("*"),),  
)  

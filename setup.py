r"""
Shim setup.py
"""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

from setuptools import setup, find_packages  
'''
from setuptools.extension import Extension
from Cython.Build import cythonize

extensions = [
    Extension("omicverse.cylib.fast_utils", ["ext_modules/fast_utils.pyx"]),
]
'''

setup(  
    name = 'omicverse',  
    version = '1.4.4',
    # keywords = ('chinesename',),  
    description = 'OmicVerse: A single pipeline for exploring the entire transcriptome universe',  
    license = 'GNU License',  
    install_requires = ['pybind11','hnswlib','ERgene','numpy','scanpy','pandas==1.5.3','matplotlib','scikit-learn','scipy','networkx','multiprocess',
                        'seaborn','datetime','statsmodels','gseapy==0.10.8','ipywidgets','lifelines','ktplotspy','python-dotplot','pybedtools>=0.8.1',
                        'boltons','ctxcore','termcolor','pygam==0.8.0','pillow','gdown','igraph','leidenalg','s_gd2','graphtools','datashader',
                        'phate','wget','tqdm','pydeseq2','mofax','adjustText',],  
    package_dir={'':'omicverse'},
    packages = find_packages("omicverse",exclude=["sample","tutorial","omicverse_guide","models",
                                      "dist"]),  # 要打包的项目文件夹
    #include_package_data=False,   # 自动打包文件夹内所有数据
    package_data={
        "omicverse": ["data_files/*.gmt"],
    },
    author = 'ZehuaZeng',  
    author_email = 'Starlitnightly@163.com',
    url = 'https://github.com/Starlitnightly/omicverse',
    long_description=long_description,  
    long_description_content_type="text/markdown",  
    python_requires='>=3.8',
    py_modules=['bulk','single','bulk2single','utils','pp'],
    #ext_modules=cythonize(extensions),
    # packages = find_packages(include=("*"),),  
)

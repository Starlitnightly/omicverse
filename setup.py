
from setuptools import setup, find_packages  

setup(  
    name = 'Pyomic',  
    version = '1.0.10',
    # keywords = ('chinesename',),  
    description = 'A python framework library for omics analysis',  
    license = 'MIT License',  
    install_requires = ['ERgene','numpy','pandas','matplotlib','sklearn','scipy','networkx','seaborn','datetime','statsmodels'],  
    packages = ['Pyomic'],  # 要打包的项目文件夹
    include_package_data=True,   # 自动打包文件夹内所有数据
    author = 'ZehuaZeng',  
    author_email = 'Starlitnightly@163.com',
    url = 'https://github.com/Starlitnightly/Pyomic',
    # packages = find_packages(include=("*"),),  
)  

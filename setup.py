# setup.py
from setuptools import setup, find_packages

setup(
    name="simple-fx",
    version="1.0.0",
    packages=find_packages(include=['src', 'src.*']),
    package_dir={'': '.'},
    install_requires=[
        'pandas>=1.5.0',
        'numpy>=1.23.0',
        'pyyaml>=6.0',
        'requests>=2.28.0',
        'pytest>=7.0.0',
    ],
    python_requires='>=3.8',
)
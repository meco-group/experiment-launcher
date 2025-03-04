from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))
requires_list = []
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    requires_list = f.read().splitlines()  # Cleaner way to read requirements

setup(
    name='experiment_launcher',
    description='Experiment Launcher',
    packages=find_packages(where="src"),  # Automatically find the correct package
    package_dir={"": "src"},  # Define 'src' as the root
    install_requires=requires_list,
)

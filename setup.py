from setuptools import setup, find_packages

setup(
    name="md_sim",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "ase",
        "mace-torch",
    ],
)
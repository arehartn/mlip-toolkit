from setuptools import setup, find_packages

setup(
    name="md_sim",
    version="0.1",
    packages=find_packages(),
    package_data={"md_sim": ["configs/*.json"]},
    include_package_data=True,
    install_requires=[
        "numpy",
        "ase",
        "mace-torch",
    ],
    entry_points={
        "console_scripts": [
            "md-sim=md_sim.run_simulation:main",
            "md-relax=md_sim.relaxation:main",
            "md-validate=md_sim.validation:main",
        ],
    },
)

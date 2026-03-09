"""Package setup."""

from setuptools import setup, find_packages

setup(
    name="dataflow",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pyyaml>=6.0",
    ],
    entry_points={
        "console_scripts": [
            "dataflow=cli.main:main",
        ],
    },
)

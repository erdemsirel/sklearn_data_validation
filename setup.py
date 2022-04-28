import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="sklearn-data-validation",
    version="0.0.1",
    author="Erdem Sirel",
    author_email="erdemsirel@gmail.com",
    description=("A tool for data validation."),
    license="BSD",
    keywords="Data Validation",
    url="https://github.com/erdemsirel/sklearn_data_validation",
    packages=["scikit-learn", "pandas", "numpy"],
    long_description=read("README"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
)
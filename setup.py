#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Setup script for DATSCAN unsupervised learning project.
"""

from setuptools import setup, find_packages

setup(
    name="datscan_unsupervised",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "scikit-learn>=1.0.0",
        "scikit-image>=0.19.0",
        "pydicom>=2.3.0",
        "nibabel>=3.2.0",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "tqdm>=4.60.0",
        "scipy>=1.7.0",
    ],
    author="Alexandre Crivellari",
    author_email="a.crivellari@campus.unimib.it",
    description="Unsupervised learning models for DATSCAN images",
    keywords="deep-learning, medical-imaging, unsupervised-learning",
    python_requires=">=3.8",
)
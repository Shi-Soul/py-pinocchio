#!/usr/bin/env python3
"""
Setup script for py-pinocchio: A fast and flexible implementation of Rigid Body Dynamics algorithms
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

# Read requirements
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="py-pinocchio",
    version="0.1.0",
    author="Shi-Soul, Augment Code",
    author_email="",
    description="A fast and flexible implementation of Rigid Body Dynamics algorithms and their analytical derivatives",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/Shi-Soul/py-pinocchio",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "examples": [
            "matplotlib>=3.0",
            "jupyter>=1.0",
        ],
    },
    # entry_points={
    #     "console_scripts": [
    #         "py-pinocchio=py_pinocchio.cli:main",
    #     ],
    # },
    include_package_data=True,
    zip_safe=False,
)

#!/usr/bin/env python3
"""
Smart Marine Project - Setup Script
===================================

Setup script for the Smart Marine Plastic Detection System.
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Smart Marine Project - Plastic Waste Detection System"

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="smart-marine-project",
    version="1.0.0",
    author="Smart Marine Project Team",
    author_email="contact@smartmarineproject.com",
    description="A comprehensive plastic waste detection system for marine environments",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/smartmarineproject/plastic-detection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Environmental Science",
    ],
    python_requires=">=3.7",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ],
        "gpu": [
            "torch-audio>=0.9.0",
            "torchaudio>=0.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "smart-marine-detect=src.plastic_detector:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["configs/*.yaml", "models/*/*.pt", "data/*.yaml"],
    },
    zip_safe=False,
)

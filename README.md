<img src="https://github.com/VilenneFrederique/CPred/blob/main/img/CPred_logo.png"
width="550" height="300" /> <br/><br/>


[![GitHub release](https://flat.badgen.net/github/release/VilenneFrederique/CPred)](https://github.com/VilenneFredericonque/CPred/releases/latest/)
[![PyPI](https://flat.badgen.net/pypi/v/cpred)](https://pypi.org/project/cpred/)
[![Conda](https://img.shields.io/conda/vn/bioconda/deeplc?style=flat-square)](https://bioconda.github.io/recipes/deeplc/README.html)
[![GitHub Workflow Status](https://flat.badgen.net/github/checks/compomics/deeplc/)](https://github.com/compomics/deeplc/actions/)
[![License](https://flat.badgen.net/github/license/VilenneFrederique/cpred)](https://www.apache.org/licenses/LICENSE-2.0)


CPred: Charge State Prediction for Modified and Unmodified Peptides in Electrospray Ionization

---

- [Introduction](#introduction)
- [Usage](#usage)
  - [Python package](#python-package)
    - [Installation](#installation)
    - [Command line interface](#command-line-interface)
    - [Python module](#python-module)
  - [Input files](#input-files)
  - [Prediction models](#prediction-models)
- [Q&A](#qa)
- [Citation](#citation)

---

## Introduction

CPred is a neural network capable of predicting the charge state distribution for
modified and unmodified peptides in electrospray ionisation. By summarising the 
modifications as measures of mass and atomic compositions, the model is capable of
generalising unseen modifications during training. 

The model is available as a Python package, installable through Pypi and conda.
This also makes it possible to use from the command-line-interface.

## How to use


### Python package


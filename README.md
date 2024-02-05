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
- [Installation](#installation)
- [How to use](#How-to-use)
  - [Python module](#Python-module)
  - [Command line interface](#command-line-interface)
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

## Installation
[![install with bioconda](https://flat.badgen.net/badge/install%20with/bioconda/green)](http://bioconda.github.io/recipes/CPred/README.html)
[![install with pip](https://flat.badgen.net/badge/install%20with/pip/green)](http://bioconda.github.io/recipes/CPred/README.html)

Install with conda, using the bioconda and conda-forge channels:
`conda install -c bioconda -c conda-forge CPred`

Or install with pip:
`pip install CPred`


## How to use
### Python module
A reproducible example is shown in the tests folder. 

```python
from CPred import FeatureEngineering
from CPred import CPred_NN
import pandas as pd

test_dictionary = {
    "Peptide_sequence": ["PEPTIDE", "EDITPEP"],
    "Modifications": ["1|Carbamidomethyl", "2|Oxidation"]
}

# Turn dictionary into a Pandas dataframe for feature engineering
test_df = pd.DataFrame(test_dictionary)

# Do feature engineering
test_features = FeatureEngineering.feature_engineering(test_df)

# Saving to parquet
test_features.to_parquet(f"tests/tests_input/test.parquet", index=False)

# Neural network predictions
input_model = "tests/tests_input/test.parquet"
model_directory = "CPred/Data/Models/CPred_model_v1.keras"
output_directory = "tests/tests_output/"
CPred_NN.prediction_model(input_model, model_directory, output_directory)
```

The feature_engineering function returns a pandas dataframe with the generated features. 
As the CPred neural network requires the data in Parquet format, it is firstly saved.

### Command-line interface
In order to use CPred from the command-line interface, run:

```sh
CPred FeatureEngineering --i <path/to/data.xlsx>
```

We highly recommend to add a peptide file with known retention times for
calibration:

```sh
CPred --file_pred  <path/to/peptide_file.csv> --file_cal <path/to/peptide_file_with_tr.csv>
```

For an overview of all CLI arguments, run `CPred --help`.

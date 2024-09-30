<img src="https://github.com/VilenneFrederique/CPred/blob/01d72bb4e2580eecf14ab63aa03ee67a31ebe235/img/CPred_logo.tif" width="550" height="300" /> <br/><br/>




[![PyPI](https://flat.badgen.net/pypi/v/cpred)](https://pypi.org/project/cpred/)
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
- [Citation](#citation)
- [Q&A](#qa)

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

Currently, CPred is solely available in Pypi. Bioconda will be released soon. 
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

#### Documentation
```sh
python CPred_main.py FeatureEngineering --help
python CPred_main.py prediction --help
python CPred_main.py retraining --help
```

#### Running
In order to run CPred, all arguments may be specificied as documented.
```sh
python CPred_main.py FeatureEngineering -i ..\tests\tests_input\test.xlsx
```

## Citation
When using CPred for your research, please cite:
Vilenne, F., Agten, A., Appeltans, S., Ertaylan, G., & Valkenborg, D. (2024). CPred: Charge State Prediction for Modified and Unmodified Peptides in Electrospray Ionization. Analytical Chemistry. https://doi.org/10.1021/acs.analchem.4c01107

## Q&A


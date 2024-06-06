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



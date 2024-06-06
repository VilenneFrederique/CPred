# Importing modules
from CPred.FeatureEngineering import feature_engineering
from CPred.CPred_NN import prediction_model, retraining_model
import pandas as pd
import os
from argparse import ArgumentParser


# Function
def main(command_line=None):
    # add main parser object
    parser = ArgumentParser(description="Charge state Prediction")

    # add sub parser object
    subparsers = parser.add_subparsers(dest="mode")
    ####################################################################################################################
    # Modelling
    Features = subparsers.add_parser("FeatureEngineering", help="Generate features.")
    # Adding arguments
    ## Input
    Features.add_argument("-i", "--input",
                          dest="input_file",
                          required=True,
                          nargs="+",
                          help='Give input data, this may either be an xlsx, csv or tsv file.')
    ## Unimod modifications
    # Features.add_argument("-m", "--modifications",
    #                      dest="modifications_list",
    #                      required=False,
    #                      default="https://raw.githubusercontent.com/VilenneFrederique/CPred/main/CPred/Data/Unimod_modifications.xlsx",
    #                      help='Potentially use your own list of modifications. (Default: Unimod_modifications.xlsx on Github')
    ## Output
    Features.add_argument("-o", "--output",
                          dest="output_file",
                          required=False,
                          default="Output_FeatureEnigineering",
                          help="If you want to save the data with features, give a filename. (Default: Output_FeatureEnigineering)")
    ## Output directory
    Features.add_argument("-d", "--directory",
                          dest="output_directory",
                          required=False,
                          default="../",
                          help="Directory for saving the results. If not provided and saving the results was set to yes, the root directory is used. (Default is in root directory)")
    ## Output format
    Features.add_argument("-f", "--format",
                          dest="output_format",
                          required=False,
                          default="parquet",
                          help="Output format for saving the results. If not provided while saving was enabled, a standard parquet file will be provided. Other formats are xlsx, csv and tsv. (Default is parquet)")

    ####################################################################################################################
    ####################################################################################################################
    # Retraining
    retraining = subparsers.add_parser("retraining", help="Retrain the CPred model")
    # Adding arguments
    ## Input
    retraining.add_argument("-i", "--input",
                            dest="input_directory",
                            required=True,
                            nargs="+",
                            help='Give input data, requires a parquet file.')
    ## Model
    retraining.add_argument("-m", "--model",
                            dest="model",
                            required=False,
                            default="CPred/Data/Models/CPred_model_v1.keras",
                            help='Which model to use')
    ## Batch size
    retraining.add_argument("-bs", "--BatchSize",
                            dest="batch_size",
                            required=True,
                            help="Batch size used to train")
    ## Learning rate
    retraining.add_argument("-lr", "--LearningRate",
                            dest="learning_rate",
                            required=True,
                            help="Learning rate used to train the model")
    ## Output
    retraining.add_argument("-o", "--output",
                            dest="output_directory",
                            required=True,
                            help="Saving the model results")

    ####################################################################################################################
    ####################################################################################################################
    # Predicting
    prediction = subparsers.add_parser("prediction",
                                       help="Predict using the CPred model")
    # Adding arguments
    ## Input
    prediction.add_argument("-i", "--input",
                            dest="input_directory",
                            required=True,
                            nargs="+",
                            help='Give input data, only accepts a parquet file.')
    ## Model
    prediction.add_argument("-m", "--model",
                            dest="model",
                            required=False,
                            default="CPred/Data/Models/CPred_model_v1.keras",
                            help='Which model to use')
    ## Output
    prediction.add_argument("-o", "--output", dest="output_directory", required=True, help="Saving the model results")
    ####################################################################################################################
    # Argument parsing
    args = parser.parse_args(command_line)
    if args.mode == "FeatureEngineering":
        input_file = args.input_file
        if input_file.endswith(".xlsx"):
            data = pd.read_excel(input_file)
            data_features = feature_engineering(dataframe=data)
            # Storing results
            output_filename = args.output_file
            output_directory = args.output_directory
            os.makedirs(output_directory, exist_ok=True)
            output_format = args.output_format
            if output_format == "parquet":
                data_features.to_parquet(f"{output_directory}{output_filename}.{output_format}", index=False)
            elif output_format == "xlsx":
                data_features.to_excel(f"{output_directory}{output_filename}.{output_format}", index=False)
            elif output_format == "csv":
                data_features.to_csv(f"{output_directory}{output_filename}.{output_format}", index=False)
            elif output_format == "xlsx":
                data_features.to_csv(f"{output_directory}{output_filename}.{output_format}", index=False, sep="\t")
            else:
                print("Unvalid file format provided!")

        elif input_file.endswith(".csv"):
            data = pd.read_csv(input_file)
            data_features = feature_engineering(dataframe=data)
            # Storing results
            output_filename = args.output_file
            output_directory = args.output_directory
            os.makedirs(output_directory, exist_ok=True)
            output_format = args.output_format
            if output_format == "parquet":
                data_features.to_parquet(f"{output_directory}{output_filename}.{output_format}", index=False)
            elif output_format == "xlsx":
                data_features.to_excel(f"{output_directory}{output_filename}.{output_format}", index=False)
            elif output_format == "csv":
                data_features.to_csv(f"{output_directory}{output_filename}.{output_format}", index=False)
            elif output_format == "xlsx":
                data_features.to_csv(f"{output_directory}{output_filename}.{output_format}", index=False, sep="\t")
            else:
                print("Unvalid file format provided!")
        elif input_file.endswith(".tsv"):
            data = pd.read_csv(input_file, sep="\t")
            data_features = feature_engineering(dataframe=data)
            # Storing results
            output_filename = args.output_file
            output_directory = args.output_directory
            os.makedirs(output_directory, exist_ok=True)
            output_format = args.output_format
            if output_format == "parquet":
                data_features.to_parquet(f"{output_directory}{output_filename}.{output_format}", index=False)
            elif output_format == "xlsx":
                data_features.to_excel(f"{output_directory}{output_filename}.{output_format}", index=False)
            elif output_format == "csv":
                data_features.to_csv(f"{output_directory}{output_filename}.{output_format}", index=False)
            elif output_format == "xlsx":
                data_features.to_csv(f"{output_directory}{output_filename}.{output_format}", index=False, sep="\t")
            else:
                print("Unvalid file format provided!")
        else:
            print(
                "Data was provided in an unsupported format. Please consult the documentation for valid formats or contact the developers to add additional formats on the corresponding GitHub page.")
            print("Feature engineering will now stop.")

    elif args.mode == "retraining":
        retraining_model(input_data=args.input_directory,
                         model_directory=args.model,
                         batch_size=args.batch_size,
                         learning_rate=args.learning_rate,
                         output_directory=args.output_directory)
    elif args.mode == "prediction":
        prediction_model(input_data=args.input_directory,
                         model_directory=args.model,
                         output_directory=args.output_directory)
    else:
        print("No action performed")
    return


if __name__ == "__main__":
    main()

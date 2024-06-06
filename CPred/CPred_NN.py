# Modules
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras import layers


def prediction_model(input_data, model_directory, output_directory):
    # Preparing data
    print("Preparing data")

    # Reading the data
    data = pd.read_parquet(input_data, engine="fastparquet")

    # Variables
    Static_variables = ["Peptide_Length", "Tryptic", "Fraction_Basic", "Fraction_Acidic",
                        "Fraction_Non_polar", "Fraction_Polar", "Fraction_Polar_basic", "Fraction_Polar_acidic",
                        "Fraction_Aromatic", "Fraction_Alpha", "Fraction_Beta", "Fraction_Turn", "Missed_Cleavages",
                        "Avg_Isoelectric_point_AA", "Avg_Hydrophobicity_AA", "N_Term_AA", "N_Term_AA_pI",
                        "N_Term_AA_pI_last_2_avg", "C_Term_AA", "C_Term_AA_pI", "C_Term_AA_pI_last_2_avg", "N", "HexN",
                        "C", "B", "Li", "Se", "K", "Zn", "NeuGc", "HexA", "Ag", "Hep", "Sulf", "18O", "Fe", "15N", "2H",
                        "S", "I", "Hex", "Ac", "Na", "O", "Hg", "F", "Ni", "As", "Br", "Mo", "Me", "Kdn", "Ca", "dHex",
                        "Pent", "Al", "HexNAc", "Cu", "H", "Cl", "P", "13C", "Mg", "NeuAc", "Monoisotopic_mass",
                        "Average_mass"]

    # Extracting the data
    ## Static variables
    data_static = data[Static_variables].to_numpy()
    ## Sequence data
    data_sequences = np.array(data["Sequences_OneHot"].tolist())
    ## Isoelectric point
    data_pI = np.array(data["Isoelectric_point_AA"].tolist())
    ## Hydrophobicity
    data_Hydrophobicity = np.array(data["Hydrophobicity_AA"].tolist())

    # Loading previous weights
    print("Loading model")
    modelLSTM = keras.models.load_model(model_directory)

    # Predictions
    predictions = modelLSTM.predict([data_sequences, data_pI, data_Hydrophobicity, data_static])

    df_predictions = pd.DataFrame(predictions)

    df_predictions.columns = ["Probability_CS+1",
                              "Probability_CS+2",
                              "Probability_CS+3",
                              "Probability_CS+4",
                              "Probability_CS+5",
                              "Probability_CS+6",
                              "Probability_CS+7"]

    df_predictions.to_csv(f"{output_directory}/Model_predictions.csv", index=False)
    return


def retraining_model(input_data, model_directory, batch_size, learning_rate, output_directory):
    # Preparing data
    print("Preparing data")

    # Reading the data
    data = pd.read_parquet(input_data, engine="fastparquet")

    # Training, test, validation split
    df_training, df_rest = train_test_split(data,
                                            train_size=0.7,
                                            shuffle=True,
                                            random_state=1996)
    df_validation, df_holdout = train_test_split(df_rest,
                                                 train_size=0.2,
                                                 shuffle=True,
                                                 random_state=1996)

    # Variables
    Outcome_variables = ["Proportion_charge_1", "Proportion_charge_2", "Proportion_charge_3", "Proportion_charge_4",
                         "Proportion_charge_5", "Proportion_charge_6", "Proportion_charge_7"]
    Static_variables = ["Peptide_Length", "Tryptic", "Fraction_Basic", "Fraction_Acidic",
                        "Fraction_Non_polar", "Fraction_Polar", "Fraction_Polar_basic", "Fraction_Polar_acidic",
                        "Fraction_Aromatic", "Fraction_Alpha", "Fraction_Beta", "Fraction_Turn", "Missed_Cleavages",
                        "Avg_Isoelectric_point_AA", "Avg_Hydrophobicity_AA", "N_Term_AA", "N_Term_AA_pI",
                        "N_Term_AA_pI_last_2_avg", "C_Term_AA", "C_Term_AA_pI", "C_Term_AA_pI_last_2_avg", "N", "HexN",
                        "C", "B", "Li", "Se", "K", "Zn", "NeuGc", "HexA", "Ag", "Hep", "Sulf", "18O", "Fe", "15N", "2H",
                        "S", "I", "Hex", "Ac", "Na", "O", "Hg", "F", "Ni", "As", "Br", "Mo", "Me", "Kdn", "Ca", "dHex",
                        "Pent", "Al", "HexNAc", "Cu", "H", "Cl", "P", "13C", "Mg", "NeuAc", "Monoisotopic_mass",
                        "Average_mass"]

    # Extracting the data
    ## Outcome
    df_training_outcome = df_training[Outcome_variables].to_numpy()
    df_validation_outcome = df_validation[Outcome_variables].to_numpy()
    ## Static variables
    df_training_static = df_training[Static_variables].to_numpy()
    df_validation_static = df_validation[Static_variables].to_numpy()
    ## Sequence data
    df_training_sequences = np.array(df_training["Sequences_OneHot"].tolist())
    df_validation_sequences = np.array(df_validation["Sequences_OneHot"].tolist())
    ## Isoelectric point
    df_training_pI = np.array(df_training["Isoelectric_point_AA"].tolist())
    df_validation_pI = np.array(df_validation["Isoelectric_point_AA"].tolist())
    ## Hydrophobicity
    df_training_Hydrophobicity = np.array(df_training["Hydrophobicity_AA"].tolist())
    df_validation_Hydrophobicity = np.array(df_validation["Hydrophobicity_AA"].tolist())

    # Input shapes
    Max_Length_Seq = 50

    # Callbacks
    callbacks_list = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, mode="min"),
                      keras.callbacks.ModelCheckpoint(filepath=f"{output_directory}/final_model_best.keras",
                                                      monitor="val_loss", save_best_only=True),
                      keras.callbacks.CSVLogger(filename=f"{output_directory}/final_model.csv", separator=",",
                                                append=False)
                      ]

    # Create the model
    ## Writing inputs
    inputs_seq = keras.Input(shape=(Max_Length_Seq,))
    inputs_isoelectric = keras.Input(shape=(Max_Length_Seq,))
    inputs_hydrophobicity = keras.Input(shape=(Max_Length_Seq,))
    inputs_static = keras.Input(shape=(df_training_static.shape[1],))

    ## LSTM Sequence part
    ### Define the embedding layer for sequences
    embedding_layer_seq = layers.Embedding(input_dim=22, output_dim=192, mask_zero=True)(inputs_seq)
    x_seq = layers.Bidirectional(layers.LSTM(units=192, recurrent_dropout=0.2))(embedding_layer_seq)

    ### Define the embedding layer for isoelectric point
    embedding_layer_isoelectric = layers.Embedding(input_dim=21, output_dim=256, mask_zero=True)(inputs_isoelectric)
    x_isoelectric = layers.Bidirectional(layers.LSTM(units=256, recurrent_dropout=0.2))(embedding_layer_isoelectric)

    ### Define the embedding layer for sequences
    embedding_layer_hydrophobicity = layers.Embedding(input_dim=21, output_dim=64, mask_zero=True)(
        inputs_hydrophobicity)
    x_hydrophobicity = layers.Bidirectional(layers.LSTM(units=64, recurrent_dropout=0.2))(
        embedding_layer_hydrophobicity)

    ## Static part
    x_static = layers.Dense(256, activation="relu")(inputs_static)
    x_static = layers.Dropout(0.2)(x_static)

    ## Concatenation
    x = layers.Concatenate(axis=1)([x_seq, x_isoelectric, x_hydrophobicity, x_static])
    x = layers.BatchNormalization()(x)

    ## Continue
    x = layers.Dense(96, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(7, activation="softmax")(x)

    modelLSTM = keras.Model(inputs=[inputs_seq, inputs_isoelectric, inputs_hydrophobicity, inputs_static],
                            outputs=outputs)

    # Compile the model
    modelLSTM.compile(optimizer=keras.optimizers.Adam(learning_rate=float(learning_rate)),
                      loss=tf.keras.losses.MeanSquaredError(), metrics=['accuracy'])

    # Printing model
    keras.utils.plot_model(modelLSTM, to_file=f"{output_directory}/Final_model_architecture.png", show_shapes=True,
                           show_layer_names=True,
                           show_layer_activations=True)

    # Loading previous weights
    print("Loading weights")
    modelLSTM.load_weights(model_directory)

    # Preparing data
    print("Continuing with training the model")

    # Train the model
    modelLSTM.fit([df_training_sequences, df_training_pI, df_training_Hydrophobicity, df_training_static],
                  df_training_outcome,
                  epochs=100,
                  batch_size=int(batch_size),
                  validation_data=(
                      [df_validation_sequences, df_validation_pI, df_validation_Hydrophobicity,
                       df_validation_static], df_validation_outcome),
                  callbacks=callbacks_list,
                  verbose=1)

    # Preparing data
    print("Saving model")

    # Saving the model
    modelLSTM.save(f"{output_directory}/final_model.keras")

    return

# Loading modules
import pandas as pd
import numpy as np
import regex as re
from tensorflow import keras


# Dictionaries
Isoelectric_point = {
    "A": 6.00,
    "C": 5.07,
    "D": 2.77,
    "E": 3.22,
    "F": 5.48,
    "G": 5.97,
    "H": 7.59,
    "I": 6.02,
    "K": 9.74,
    "L": 5.98,
    "M": 5.74,
    "N": 5.41,
    "P": 6.30,
    "Q": 5.65,
    "R": 10.76,
    "S": 5.68,
    "T": 5.60,
    "V": 5.96,
    "W": 5.89,
    "Y": 5.66,
    "O": 9.74
}

Hydrophobicity = {      # Black et al. 1991 https://doi.org/10.1016/0003-2697(91)90045-U
    "A": 0.616,
    "C": 0.680,
    "D": 0.028,
    "E": 0.043,
    "F": 1.000,
    "G": 0.501,
    "H": 0.165,
    "I": 0.943,
    "K": 0.283,
    "L": 0.943,
    "M": 0.738,
    "N": 0.236,
    "P": 0.711,
    "Q": 0.251,
    "R": 0.00001,
    "S": 0.359,
    "T": -0.70,
    "V": 0.825,
    "W": 0.878,
    "Y": 0.880,
    "O": 0.283
}

Sequence_One_Hot = {'A': 1,
                    'C': 2,
                    'D': 3,
                    'E': 4,
                    'F': 5,
                    'G': 6,
                    'H': 7,
                    'I': 8,
                    'K': 9,
                    'L': 10,
                    'M': 11,
                    'N': 12,
                    'P': 13,
                    'Q': 14,
                    'R': 15,
                    'S': 16,
                    'T': 17,
                    'V': 18,
                    'W': 19,
                    'Y': 20,
                    'O': 21}

Charge_State_OneHot = {
    1: [1, 0, 0, 0, 0, 0],
    2: [0, 1, 0, 0, 0, 0],
    3: [0, 0, 1, 0, 0, 0],
    4: [0, 0, 0, 1, 0, 0],
    5: [0, 0, 0, 0, 1, 0],
    6: [0, 0, 0, 0, 0, 1]
}

ElementalComp = {
    'Amino Acid': ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'O'],
    'C': [3, 6, 4, 4, 3, 5, 5, 2, 6, 6, 6, 6, 5, 9, 5, 3, 4, 11, 9, 5, 12],
    'H': [5, 12, 6, 5, 5, 7, 8, 3, 7, 11, 11, 12, 9, 9, 7, 5, 7, 10, 9, 9, 21],
    'N': [1, 4, 2, 1, 1, 1, 2, 1, 3, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 3],
    'O': [1, 1, 2, 3, 1, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 2, 1, 3],
    'S': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
}
ElementalComp = pd.DataFrame(ElementalComp)
ElementalComp = ElementalComp.sort_values(by='Amino Acid')
ElementalComp.set_index('Amino Acid', inplace=True)
ElementalComp.index.name = 'Amino Acid'


# All functions related to feature engineering
def tryptic(dataframe):
    dataframe["Tryptic"] = list(map(lambda x: x.endswith('R') | x.endswith('K'), dataframe['Peptide_sequence']))
    return dataframe


def peptide_length(dataframe):
    dataframe["Peptide_Length"] = dataframe["Peptide_sequence"].str.len()
    return dataframe


def basic(dataframe):
    dataframe["Fraction_Basic"] = dataframe['Peptide_sequence'].str.count('[RHKO]') / dataframe['Peptide_Length']
    return dataframe


def acid(dataframe):
    dataframe["Fraction_Acidic"] = dataframe['Peptide_sequence'].str.count('[DE]') / dataframe['Peptide_Length']
    return dataframe


def non_polar(dataframe):
    dataframe["Fraction_Non_polar"] = dataframe['Peptide_sequence'].str.count('[AFGILMPVW]') / dataframe['Peptide_Length']
    return dataframe


def polar(dataframe):
    dataframe["Fraction_Polar"] = dataframe['Peptide_sequence'].str.count('[CNQSTY]') / dataframe['Peptide_Length']
    return dataframe


def polar_basic(dataframe):
    dataframe["Fraction_Polar_basic"] = dataframe['Peptide_sequence'].str.count('[HKRO]') / dataframe['Peptide_Length']
    return dataframe


def polar_acidic(dataframe):
    dataframe["Fraction_Polar_acidic"] = dataframe['Peptide_sequence'].str.count('[DE]') / dataframe['Peptide_Length']
    return dataframe


def aromatic(dataframe):
    dataframe["Fraction_Aromatic"] = dataframe['Peptide_sequence'].str.count('[FWY]') / dataframe['Peptide_Length']
    return dataframe


def alpha(dataframe):
    dataframe["Fraction_Alpha"] = dataframe['Peptide_sequence'].str.count('[FILVWY]') / dataframe['Peptide_Length']
    return dataframe


def beta(dataframe):
    dataframe["Fraction_Beta"] = dataframe['Peptide_sequence'].str.count('[GNPS]') / dataframe['Peptide_Length']
    return dataframe


def turn(dataframe):
    dataframe["Fraction_Turn"] = dataframe['Peptide_sequence'].str.count('[AELM]') / dataframe['Peptide_Length']
    return dataframe


def sequence_checker(sequence):
    if sequence.endswith('R') | sequence.endswith('K'):
        Arginines = len(re.findall(pattern='R', string=sequence))
        Lysines = len(re.findall(pattern='K', string=sequence))
        Inhibited_RP = len(re.findall(pattern='RP', string=sequence))
        Inhibited_KP = len(re.findall(pattern='KP', string=sequence))
        Missed_Cleavage = Arginines + Lysines - Inhibited_RP - Inhibited_KP - 1
        return Missed_Cleavage
    else:
        Missed_Cleavage = 0
        return Missed_Cleavage


def missed_cleavages(dataframe):
    dataframe["Missed_Cleavages"] = dataframe['Peptide_sequence'].apply(lambda x: sequence_checker(x))
    return dataframe


def isoelectric_list(sequence):
    storage = []
    for AA in sequence:
        isoelectric_value = Isoelectric_point[AA]
        storage.append(isoelectric_value)
    return storage


def isoelectric_point(dataframe):
    dataframe["Isoelectric_point_AA"] = dataframe['Peptide_sequence'].apply(lambda x: isoelectric_list(x))
    return dataframe


def average_isoelectric_point(dataframe):
    dataframe["Avg_Isoelectric_point_AA"] = dataframe['Isoelectric_point_AA'].apply(lambda x: np.mean(x))
    return dataframe


def hydrophobicity_list(sequence):
    storage = []
    for AA in sequence:
        Hydrophobicity_value = Hydrophobicity[AA]
        storage.append(Hydrophobicity_value)
    return storage


def hydrophobicity(dataframe):
    dataframe["Hydrophobicity_AA"] = dataframe['Peptide_sequence'].apply(lambda x: hydrophobicity_list(x))
    return dataframe


def average_hydrophobicity(dataframe):
    dataframe["Avg_Hydrophobicity_AA"] = dataframe['Hydrophobicity_AA'].apply(lambda x: np.mean(x))
    return dataframe


def elemental_comp_list(sequence, element):
    element_count = 0
    for amino_acid in sequence:
        element_count += ElementalComp.loc[amino_acid][element]
    # Add water at the end
    if element == "H":
        element_count += 2
    elif element == "O":
        element_count += 1
    return element_count


Unique_elements = ['N', 'HexN', 'C', 'B', 'Li', 'Se', 'K', 'Zn', 'NeuGc', 'HexA', 'Ag', 'Hep', 'Sulf', '18O', 'Fe', '15N',
                    '2H', 'S', 'I', 'Hex', 'Ac', 'Na', 'O', 'Hg', 'F', 'Ni', 'As', 'Br', 'Mo', 'Me', 'Kdn', 'Ca', 'dHex', 'Pent',
                    'Al', 'HexNAc', 'Cu', 'H', 'Cl', 'P', '13C', 'Mg', 'NeuAc']


def elemental_comp(dataframe):
    for column in Unique_elements:
        dataframe[column] = 0
    dataframe["C"] = dataframe.apply(lambda x: elemental_comp_list(x['Peptide_sequence'], "C"), axis=1)
    dataframe["H"] = dataframe.apply(lambda x: elemental_comp_list(x['Peptide_sequence'], "H"), axis=1)
    dataframe["O"] = dataframe.apply(lambda x: elemental_comp_list(x['Peptide_sequence'], "O"), axis=1)
    dataframe["N"] = dataframe.apply(lambda x: elemental_comp_list(x['Peptide_sequence'], "N"), axis=1)
    dataframe["S"] = dataframe.apply(lambda x: elemental_comp_list(x['Peptide_sequence'], "S"), axis=1)
    return dataframe


def apply_modifications(row):
    # Reading data
    modifications_unimod = pd.read_excel(
        "https://raw.githubusercontent.com/VilenneFrederique/CPred/master/CPred/Data/Unimod_modifications.xlsx")
    # modifications_unimod = pd.read_excel("CPred/Data/Unimod_modifications.xlsx")

    modifications_string = row['Modifications']
    if pd.isna(modifications_string):
        pass
    else:
        modifications_list = re.findall(pattern="\|([\da-zA-Z]+)", string=modifications_string)
        for modification in modifications_list:
            modification_unimod_string = modifications_unimod[modifications_unimod['name'] == modification]
            for element in Unique_elements:
                row[element] += modification_unimod_string[element].values[0]
    return row


def elemental_comp_modifications(dataframe):
    dataframe_modified = dataframe.apply(apply_modifications, axis=1)
    return dataframe_modified


def N_term_AA_Checker(sequence):
    N_term_AA_sequence = sequence[0]
    One_hot_N_Term_AA = Sequence_One_Hot[N_term_AA_sequence]
    return One_hot_N_Term_AA


def N_term_AA(dataframe):
    dataframe["N_Term_AA"] = dataframe['Peptide_sequence'].apply(lambda x: N_term_AA_Checker(x))
    return dataframe


def C_term_AA_Checker(sequence):
    C_term_AA_sequence = sequence[-1]
    One_hot_C_Term_AA = Sequence_One_Hot[C_term_AA_sequence]
    return One_hot_C_Term_AA


def C_term_AA(dataframe):
    dataframe["C_Term_AA"] = dataframe['Peptide_sequence'].apply(lambda x: C_term_AA_Checker(x))
    return dataframe


def N_term_AA_pI_checker(sequence):
    N_term_AA_sequence = sequence[0]
    pI_N_Term_AA = Isoelectric_point[N_term_AA_sequence]
    return pI_N_Term_AA


def N_term_AA_pI(dataframe):
    dataframe["N_Term_AA_pI"] = dataframe['Peptide_sequence'].apply(lambda x: N_term_AA_pI_checker(x))
    return dataframe


def N_term_AA_pI_checker_last_2(sequence):
    N_term_AA_sequence_AA_0 = sequence[0]
    N_term_AA_sequence_AA_1 = sequence[1]
    pI_N_Term_AA_0 = Isoelectric_point[N_term_AA_sequence_AA_0]
    pI_N_Term_AA_1 = Isoelectric_point[N_term_AA_sequence_AA_1]
    pI_N_AVG = (pI_N_Term_AA_0 + pI_N_Term_AA_1) / 2
    return pI_N_AVG


def N_term_AA_pI_last_2(dataframe):
    dataframe["N_Term_AA_pI_last_2_avg"] = dataframe['Peptide_sequence'].apply(lambda x: N_term_AA_pI_checker_last_2(x))
    return dataframe


def C_term_AA_pI_checker(sequence):
    C_term_AA_sequence = sequence[-1]
    pI_C_Term_AA = Isoelectric_point[C_term_AA_sequence]
    return pI_C_Term_AA


def C_term_AA_pI(dataframe):
    dataframe["C_Term_AA_pI"] = dataframe['Peptide_sequence'].apply(lambda x: C_term_AA_pI_checker(x))
    return dataframe


def C_term_AA_pI_checker_last_2(sequence):
    C_term_AA_sequence_AA_0 = sequence[-1]
    C_term_AA_sequence_AA_1 = sequence[-2]
    pI_C_Term_AA_0 = Isoelectric_point[C_term_AA_sequence_AA_0]
    pI_C_Term_AA_1 = Isoelectric_point[C_term_AA_sequence_AA_1]
    pI_C_AVG = (pI_C_Term_AA_0 + pI_C_Term_AA_1) / 2
    return pI_C_AVG


def C_term_AA_pI_last_2(dataframe):
    dataframe["C_Term_AA_pI_last_2_avg"] = dataframe['Peptide_sequence'].apply(lambda x: C_term_AA_pI_checker_last_2(x))
    return dataframe


def monoisotopic_mass(dataframe):
    dataframe["Monoisotopic_mass"] = np.nan
    for index in range(len(dataframe)):
        mass = (dataframe.loc[index, "N"] * 14.003074 + dataframe.loc[index, "HexN"] * 161.068808 + dataframe.loc[index, "C"] * 12 + dataframe.loc[index, "B"] * 11.0093055 + dataframe.loc[index, "Li"] * 7.016003 +
                dataframe.loc[index, "Se"] * 79.9165196 + dataframe.loc[index, "K"] * 38.9637074 + dataframe.loc[index, "Zn"] * 63.9291448 + dataframe.loc[index, "NeuGc"] * 307.090331 + dataframe.loc[index, "HexA"] * 176.032088 +
                dataframe.loc[index, "Ag"] * 106.905092 + dataframe.loc[index, "Hep"] * 192.063388 + dataframe.loc[index, "Sulf"] * 31.9720707 + dataframe.loc[index, "18O"] * 17.9991603 + dataframe.loc[index, "Fe"] * 55.9349393 +
                dataframe.loc[index, "15N"] * 15.00010897 + dataframe.loc[index, "2H"] * 2.014101779 + dataframe.loc[index, "S"] * 31.9720707 + dataframe.loc[index, "I"] * 126.904473 + dataframe.loc[index, "Hex"] * 162.052824 +
                dataframe.loc[index, "Ac"] * 59.013851 + dataframe.loc[index, "Na"] * 22.9897677 + dataframe.loc[index, "O"] * 15.99491463 + dataframe.loc[index, "Hg"] * 201.970617 + dataframe.loc[index, "F"] * 18.99840322 +
                dataframe.loc[index, "Ni"] * 57.9353462 + dataframe.loc[index, "As"] * 74.9215942 + dataframe.loc[index, "Br"] * 78.9183361 + dataframe.loc[index, "Mo"] * 97.9054073 + dataframe.loc[index, "Me"] * 15.023475105 +
                dataframe.loc[index, "Kdn"] * 268.079437 + dataframe.loc[index, "Ca"] * 39.9625906 + dataframe.loc[index, "dHex"] * 146.057909 + dataframe.loc[index, "Pent"] * 132.042259 + dataframe.loc[index, "Al"] * 26.9815386 +
                dataframe.loc[index, "HexNAc"] * 203.079373 + dataframe.loc[index, "Cu"] * 62.9295989 + dataframe.loc[index, "H"] * 1.007825035 + dataframe.loc[index, "Cl"] * 34.96885272 + dataframe.loc[index, "P"] * 30.973762 +
                dataframe.loc[index, "13C"] * 13.00335483 + dataframe.loc[index, "Mg"] * 23.9850423 + dataframe.loc[index, "NeuAc"] * 291.095417)

        dataframe.loc[index, "Monoisotopic_mass"] = mass

    return dataframe


def average_mass(dataframe):
    dataframe["Average_mass"] = np.nan
    for index in range(len(dataframe)):
        mass = (dataframe.loc[index, "N"] * 14.0067 + dataframe.loc[index, "HexN"] * 161.1558 + dataframe.loc[index, "C"] * 12.0107 + dataframe.loc[index, "B"] * 10.811 + dataframe.loc[index, "Li"] * 6.941 +
                dataframe.loc[index, "Se"] * 78.96 + dataframe.loc[index, "K"] * 39.0983 + dataframe.loc[index, "Zn"] * 65.409 + dataframe.loc[index, "NeuGc"] * 307.2540 + dataframe.loc[index, "HexA"] * 176.1241 +
                dataframe.loc[index, "Ag"] * 107.8682 + dataframe.loc[index, "Hep"] * 192.1666 + dataframe.loc[index, "Sulf"] * 32.065 + dataframe.loc[index, "18O"] * 17.9991603 + dataframe.loc[index, "Fe"] * 55.845 +
                dataframe.loc[index, "15N"] * 15.00010897 + dataframe.loc[index, "2H"] * 2.014101779 + dataframe.loc[index, "S"] * 32.065 + dataframe.loc[index, "I"] * 126.90447 + dataframe.loc[index, "Hex"] * 162.1406 +
                dataframe.loc[index, "Ac"] * 59.045 + dataframe.loc[index, "Na"] * 22.98977 + dataframe.loc[index, "O"] * 15.9994 + dataframe.loc[index, "Hg"] * 200.59 + dataframe.loc[index, "F"] * 18.9984032 +
                dataframe.loc[index, "Ni"] * 58.6934 + dataframe.loc[index, "As"] * 74.9215942 + dataframe.loc[index, "Br"] * 79.904 + dataframe.loc[index, "Mo"] * 95.94 + dataframe.loc[index, "Me"] * 15.03452 +
                dataframe.loc[index, "Kdn"] * 268.218 + dataframe.loc[index, "Ca"] * 40.078 + dataframe.loc[index, "dHex"] * 146.1412 + dataframe.loc[index, "Pent"] * 132.1146 + dataframe.loc[index, "Al"] * 26.9815386 +
                dataframe.loc[index, "HexNAc"] * 203.1925 + dataframe.loc[index, "Cu"] * 63.546 + dataframe.loc[index, "H"] * 1.00794 + dataframe.loc[index, "Cl"] * 35.453 + dataframe.loc[index, "P"] * 30.973761 +
                dataframe.loc[index, "13C"] * 13.00335483 + dataframe.loc[index, "Mg"] * 24.305 + dataframe.loc[index, "NeuAc"] * 291.2546)

        dataframe.loc[index, "Average_mass"] = mass

    return dataframe

def encoder_sequence(sequence):
    encoded_sequence = [Sequence_One_Hot[aa] for aa in sequence]
    return encoded_sequence


def sequence_onehot(dataframe):
    dataframe["Sequences_OneHot"] = dataframe.apply(lambda x: encoder_sequence(x["Peptide_sequence"]), axis=1)
    Encoded_sequences = dataframe["Sequences_OneHot"].to_numpy()
    Encoded_sequences = keras.preprocessing.sequence.pad_sequences(Encoded_sequences,
                                                                   maxlen=50,
                                                                   padding='post',
                                                                   truncating='post')
    dataframe["Sequences_OneHot"] = Encoded_sequences.tolist()
    return dataframe


def pad_values(dataframe, column_name):
    values = dataframe[column_name].to_numpy()
    padded_input = keras.preprocessing.sequence.pad_sequences(values,
                                                              maxlen=50,
                                                              padding='post',
                                                              truncating='post',
                                                              dtype='float32')
    dataframe[column_name] = padded_input.tolist()
    dataframe[column_name] = dataframe[column_name].apply(lambda x: [(round(value, 2)) for value in x])
    return dataframe


def feature_engineering(dataframe):
    print("Starting with Feature engineering")
    dataframe = peptide_length(dataframe)
    dataframe = tryptic(dataframe)
    dataframe["Tryptic"] = dataframe["Tryptic"].replace([True, False], [1, 0])
    dataframe = basic(dataframe)
    dataframe = acid(dataframe)
    dataframe = non_polar(dataframe)
    dataframe = polar(dataframe)
    dataframe = polar_basic(dataframe)
    dataframe = polar_acidic(dataframe)
    dataframe = aromatic(dataframe)
    dataframe = alpha(dataframe)
    dataframe = beta(dataframe)
    dataframe = turn(dataframe)
    dataframe = missed_cleavages(dataframe)
    dataframe = isoelectric_point(dataframe)
    dataframe = average_isoelectric_point(dataframe)
    dataframe = hydrophobicity(dataframe)
    dataframe = average_hydrophobicity(dataframe)
    dataframe = sequence_onehot(dataframe)
    dataframe = pad_values(dataframe, "Isoelectric_point_AA")
    dataframe = pad_values(dataframe, "Hydrophobicity_AA")
    dataframe = N_term_AA(dataframe)
    dataframe = C_term_AA(dataframe)
    dataframe = N_term_AA_pI(dataframe)
    dataframe = N_term_AA_pI_last_2(dataframe)
    dataframe = C_term_AA_pI(dataframe)
    dataframe = C_term_AA_pI_last_2(dataframe)
    dataframe = elemental_comp(dataframe)
    dataframe = elemental_comp_modifications(dataframe)
    dataframe = monoisotopic_mass(dataframe)
    dataframe = average_mass(dataframe)
    print("Finished with Feature engineering")
    return dataframe

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
    return list(map(Isoelectric_point.__getitem__, sequence))


def isoelectric_point(dataframe):
    dataframe["Isoelectric_point_AA"] = dataframe['Peptide_sequence'].apply(lambda x: isoelectric_list(x))
    return dataframe


def average_isoelectric_point(dataframe):
    dataframe["Avg_Isoelectric_point_AA"] = dataframe['Isoelectric_point_AA'].apply(lambda x: np.mean(x))
    return dataframe


def hydrophobicity_list(sequence):
    return list(map(Hydrophobicity.__getitem__, sequence))


def hydrophobicity(dataframe):
    dataframe["Hydrophobicity_AA"] = dataframe['Peptide_sequence'].apply(lambda x: hydrophobicity_list(x))
    return dataframe


def average_hydrophobicity(dataframe):
    dataframe["Avg_Hydrophobicity_AA"] = dataframe['Hydrophobicity_AA'].apply(lambda x: np.mean(x))
    return dataframe


Unique_elements = ['N', 'HexN', 'C', 'B', 'Li', 'Se', 'K', 'Zn', 'NeuGc', 'HexA', 'Ag', 'Hep', 'Sulf', '18O', 'Fe', '15N',
                    '2H', 'S', 'I', 'Hex', 'Ac', 'Na', 'O', 'Hg', 'F', 'Ni', 'As', 'Br', 'Mo', 'Me', 'Kdn', 'Ca', 'dHex', 'Pent',
                    'Al', 'HexNAc', 'Cu', 'H', 'Cl', 'P', '13C', 'Mg', 'NeuAc']

# Amino acids present in the elemental composition table, used to count residue occurrences per sequence
Peptide_elements = ['C', 'H', 'O', 'N', 'S']
Amino_acids = ElementalComp.index.tolist()
ElementalComp_matrix = ElementalComp.loc[Amino_acids, Peptide_elements].to_numpy()


def elemental_comp(dataframe):
    for column in Unique_elements:
        dataframe[column] = 0
    # Count occurrences of each amino acid per sequence (vectorized), then derive C/H/O/N/S via a single matrix
    # multiplication instead of a per-residue DataFrame lookup
    amino_acid_counts = np.column_stack(
        [dataframe['Peptide_sequence'].str.count(amino_acid).to_numpy() for amino_acid in Amino_acids]
    )
    element_sums = amino_acid_counts @ ElementalComp_matrix
    element_sums = pd.DataFrame(element_sums, columns=Peptide_elements, index=dataframe.index)
    # Add water at the end
    element_sums["H"] += 2
    element_sums["O"] += 1
    dataframe[Peptide_elements] = element_sums
    return dataframe


def load_unimod_modifications():
    # Reading data (loaded once and reused for every row, instead of once per row)
    modifications_unimod = pd.read_excel(
        "https://raw.githubusercontent.com/VilenneFrederique/CPred/master/CPred/Data/Unimod_modifications.xlsx")
    # modifications_unimod = pd.read_excel("CPred/Data/Unimod_modifications.xlsx")

    # Some modification names occur more than once in the sheet; keep the first occurrence to match prior behaviour
    return modifications_unimod.drop_duplicates(subset="name", keep="first").set_index("name")


def apply_modifications(row, modifications_unimod):
    modifications_string = row['Modifications']
    if pd.isna(modifications_string):
        pass
    else:
        modifications_list = re.findall(pattern="\|([\da-zA-Z]+)", string=modifications_string)
        for modification in modifications_list:
            modification_unimod_row = modifications_unimod.loc[modification]
            for element in Unique_elements:
                row[element] += modification_unimod_row[element]
    return row


def elemental_comp_modifications(dataframe):
    modifications_unimod = load_unimod_modifications()
    dataframe_modified = dataframe.apply(lambda row: apply_modifications(row, modifications_unimod), axis=1)
    return dataframe_modified


def N_term_AA(dataframe):
    dataframe["N_Term_AA"] = dataframe['Peptide_sequence'].str[0].map(Sequence_One_Hot)
    return dataframe


def C_term_AA(dataframe):
    dataframe["C_Term_AA"] = dataframe['Peptide_sequence'].str[-1].map(Sequence_One_Hot)
    return dataframe


def N_term_AA_pI(dataframe):
    dataframe["N_Term_AA_pI"] = dataframe['Peptide_sequence'].str[0].map(Isoelectric_point)
    return dataframe


def N_term_AA_pI_last_2(dataframe):
    pI_AA_0 = dataframe['Peptide_sequence'].str[0].map(Isoelectric_point)
    pI_AA_1 = dataframe['Peptide_sequence'].str[1].map(Isoelectric_point)
    dataframe["N_Term_AA_pI_last_2_avg"] = (pI_AA_0 + pI_AA_1) / 2
    return dataframe


def C_term_AA_pI(dataframe):
    dataframe["C_Term_AA_pI"] = dataframe['Peptide_sequence'].str[-1].map(Isoelectric_point)
    return dataframe


def C_term_AA_pI_last_2(dataframe):
    pI_AA_0 = dataframe['Peptide_sequence'].str[-1].map(Isoelectric_point)
    pI_AA_1 = dataframe['Peptide_sequence'].str[-2].map(Isoelectric_point)
    dataframe["C_Term_AA_pI_last_2_avg"] = (pI_AA_0 + pI_AA_1) / 2
    return dataframe


# Element -> monoisotopic/average mass weight, used as a vector for a single dot-product per dataframe
# instead of a per-row, per-element scalar lookup loop
Monoisotopic_weights = {
    "N": 14.003074, "HexN": 161.068808, "C": 12, "B": 11.0093055, "Li": 7.016003,
    "Se": 79.9165196, "K": 38.9637074, "Zn": 63.9291448, "NeuGc": 307.090331, "HexA": 176.032088,
    "Ag": 106.905092, "Hep": 192.063388, "Sulf": 31.9720707, "18O": 17.9991603, "Fe": 55.9349393,
    "15N": 15.00010897, "2H": 2.014101779, "S": 31.9720707, "I": 126.904473, "Hex": 162.052824,
    "Ac": 59.013851, "Na": 22.9897677, "O": 15.99491463, "Hg": 201.970617, "F": 18.99840322,
    "Ni": 57.9353462, "As": 74.9215942, "Br": 78.9183361, "Mo": 97.9054073, "Me": 15.023475105,
    "Kdn": 268.079437, "Ca": 39.9625906, "dHex": 146.057909, "Pent": 132.042259, "Al": 26.9815386,
    "HexNAc": 203.079373, "Cu": 62.9295989, "H": 1.007825035, "Cl": 34.96885272, "P": 30.973762,
    "13C": 13.00335483, "Mg": 23.9850423, "NeuAc": 291.095417,
}

Average_weights = {
    "N": 14.0067, "HexN": 161.1558, "C": 12.0107, "B": 10.811, "Li": 6.941,
    "Se": 78.96, "K": 39.0983, "Zn": 65.409, "NeuGc": 307.2540, "HexA": 176.1241,
    "Ag": 107.8682, "Hep": 192.1666, "Sulf": 32.065, "18O": 17.9991603, "Fe": 55.845,
    "15N": 15.00010897, "2H": 2.014101779, "S": 32.065, "I": 126.90447, "Hex": 162.1406,
    "Ac": 59.045, "Na": 22.98977, "O": 15.9994, "Hg": 200.59, "F": 18.9984032,
    "Ni": 58.6934, "As": 74.9215942, "Br": 79.904, "Mo": 95.94, "Me": 15.03452,
    "Kdn": 268.218, "Ca": 40.078, "dHex": 146.1412, "Pent": 132.1146, "Al": 26.9815386,
    "HexNAc": 203.1925, "Cu": 63.546, "H": 1.00794, "Cl": 35.453, "P": 30.973761,
    "13C": 13.00335483, "Mg": 24.305, "NeuAc": 291.2546,
}


def monoisotopic_mass(dataframe):
    weights = pd.Series(Monoisotopic_weights)
    dataframe["Monoisotopic_mass"] = dataframe[weights.index].to_numpy() @ weights.to_numpy()
    return dataframe


def average_mass(dataframe):
    weights = pd.Series(Average_weights)
    dataframe["Average_mass"] = dataframe[weights.index].to_numpy() @ weights.to_numpy()
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

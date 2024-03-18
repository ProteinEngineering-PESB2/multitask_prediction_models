import pandas as pd
from joblib import Parallel, delayed
from lib.constant_values import constant_values

class physicochemical_encoder(object):

    def __init__(self,
                 dataset,
                 property_encoder,
                 dataset_encoder,
                 name_column_seq,
                 columns_to_ignore):

        self.dataset = dataset
        self.property_encoder = property_encoder
        self.dataset_encoder = dataset_encoder
        self.constant_instance = constant_values()
        self.name_column_seq = name_column_seq
        self.columns_to_ignore = columns_to_ignore

        self.zero_padding = self.check_max_size()

    def __check_residues(self, residue):
        if residue in self.constant_instance.possible_residues:
            return True
        else:
            return False

    def __encoding_residue(self, residue):

        if self.__check_residues(residue):
            return self.dataset_encoder[self.property_encoder][residue]
        else:
            return False

    def check_max_size(self):
        size_list = [len(seq) for seq in self.dataset[self.name_column_seq]]
        return max(size_list)

    def __encoding_sequence(self, sequence):

        sequence = sequence.upper()
        sequence_encoding = []

        for i in range(len(sequence)):
            residue = sequence[i]
            response_encoding = self.__encoding_residue(residue)
            if response_encoding != False:
                sequence_encoding.append(response_encoding)

        # complete zero padding
        for k in range(len(sequence_encoding), self.zero_padding):
            sequence_encoding.append(0)

        return sequence_encoding

    def encoding_dataset(self):

        matrix_data = []

        for index in self.dataset.index:
            coded_sequence = self.__encoding_sequence(self.dataset[self.name_column_seq][index])
            matrix_data.append(coded_sequence)

        print("Creating dataset")
        header = ['p_{}'.format(i) for i in range(len(matrix_data[0]))]
        df_data = pd.DataFrame(matrix_data, columns=header)

        for column in self.columns_to_ignore:
            df_data[column] = self.dataset[column]

        print("Export dataset")
        return df_data

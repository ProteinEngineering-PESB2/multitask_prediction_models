import math
from scipy.fft import fft
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lib.constant_values import constant_values

class fft_encoding(object):

    def __init__(self, dataset, size_data, column_with_values):
        self.dataset = dataset
        self.size_data = size_data
        self.column_with_values = column_with_values
        self.constant_instance = constant_values()

        self.init_process()

    def __processing_data_to_fft(self):

        print("Processin data")
        self.df_columns_ignore = self.dataset[self.column_with_values]
        self.dataset = self.dataset.drop(columns=self.column_with_values)

    def __get_near_pow(self):

        print("Get near pow 2 value")
        list_data = [math.pow(2, i) for i in range(1, 20)]
        stop_value = list_data[0]

        for value in list_data:
            if value >= self.size_data:
                stop_value = value
                break

        self.stop_value = int(stop_value)

    def __complete_zero_padding(self):

        print("Apply zero padding")
        list_df = [self.dataset]
        for i in range(self.size_data, self.stop_value):
            column = [0 for k in range(len(self.dataset))]
            key_name = "p_{}".format(i)
            df_tmp = pd.DataFrame()
            df_tmp[key_name] = column
            list_df.append(df_tmp)

        self.dataset = pd.concat(list_df, axis=1)

    def init_process(self):
        self.__processing_data_to_fft()
        self.__get_near_pow()
        self.__complete_zero_padding()

    def __create_row(self, index):
        row = [self.dataset[column][index] for column in self.dataset.columns]
        return row

    def __apply_FFT(self, index):

        row = self.__create_row(index)
        T = 1.0 / float(self.stop_value)
        yf = fft(row)
        xf = np.linspace(0.0, 1.0 / (2.0 * T), self.stop_value // 2)
        yf = np.abs(yf[0:self.stop_value // 2])
        return [value for value in yf]

    def encoding_dataset(self):

        print("Start FFT encoding process")
        matrix_data = []

        for index in self.dataset.index:
            row_coded = self.__apply_FFT(index)
            matrix_data.append(row_coded)

        print("Creating dataset")
        header = ['p_{}'.format(i) for i in range(len(matrix_data[0]))]
        print("Export dataset")
        df_data = pd.DataFrame(matrix_data, columns=header)
        
        df_data = pd.concat([df_data, self.df_columns_ignore], axis=1)

        return df_data
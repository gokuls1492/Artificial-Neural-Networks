import pandas as pd
import numpy as np
import os.path
import sys
#**************Authors: Gokul Surendra & Padma Kurdgi*******************************************
def read_file(filepath):
    dataframe = pd.read_csv(filepath, header=-1)
    dataframe = dataframe.dropna()  #Removes missing values
    return dataframe

#Converts column to categorical
def convert_to_categorical(feature, column):
    dataset[column] = pd.Categorical(dataset[column]).codes

#Scaling: Subtracting the mean from each of the values and dividing by the standard deviation.
def scale_feature(data, column):
    feature = data[column]
    mean_val = np.mean(feature)
    sd_val = np.std(feature)
    for index, row in data.iterrows():
        new_val = (row[column] - mean_val) / sd_val
        dataset.set_value(index, column, new_val)


input_path = sys.argv[1]#'D:\\UTD\\ML\\Assignment3\\Iris_data.txt'
output_path = sys.argv[2]#'D:\\UTD\\ML\\Assignment3\\input_prep.csv'

dataset = read_file(input_path)
path = os.path.join(output_path)
output_file = open(path, "w")

for col in dataset:
    if dataset[col].dtype == 'object':
        convert_to_categorical(dataset[col], col)

for col in dataset:
    if (dataset[col].dtype == 'float64' or 'int64') and col != len(dataset.columns)-1:
        scale_feature(dataset.iloc[:,:-1], col)

dataset.to_csv(output_file, encoding='utf-8', index=False, header=False)
output_file.close()
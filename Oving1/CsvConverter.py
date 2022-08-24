import pandas as pd



def get_tensor_from_csv(path):
    pd.read_csv(path, delimiter=',', header=None, names=['length', 'weight'])

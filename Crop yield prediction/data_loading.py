# data_loading.py
import pandas as pd
resource_folder = 'https://raw.githubusercontent.com/Rophael/Yield-Prediction/main/res'
pesticides_url = '{}/pesticides.csv?token=GHSAT0AAAAAAB7BDQDOI6CDTNFNA6UTWZJEZAGII2A'.format(resource_folder)
rainfall_url = '{}/rainfall.csv?token=GHSAT0AAAAAAB7BDQDOL46F4W4CG5RMDI6AZAGIJIQ'.format(resource_folder)
temperature_url = '{}/temp.csv?token=GHSAT0AAAAAAB7BDQDOQSWN4HP3KL5KGTAWZAGIJYQ'.format(resource_folder)
yield_url = '{}/yield.csv?token=GHSAT0AAAAAAB7BDQDPGS2GJV2NUFI3IUHOZAGIKHA'.format(resource_folder)

def load_pesticides_data():
    return pd.read_csv(pesticides_url, sep=',')

def load_rainfall_data():
    return pd.read_csv(rainfall_url, sep=', ')

def load_temperature_data():
    return pd.read_csv(temperature_url, sep=', ')

def load_yield_data():
    return pd.read_csv(yield_url, sep=',')

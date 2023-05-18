# data_preparation.py

import pandas as pd

def prepare_rainfall_data(df):
    df = df.groupby(['Year', 'Country', 'ISO3'], as_index=False, axis=0).sum()
    return df.drop(['Statistics'], axis=1)

def prepare_temperature_data(df):
    return df.groupby(['Year', 'Country', 'ISO3'], as_index=False, axis=0).mean(numeric_only=True)

def prepare_yield_data(df):
    df = df.drop(['Domain', 'Element'], axis=1)
    df.rename({'Area': 'Country', 'Value': 'Yield (hg/ha)'}, axis=1, inplace=True)
    return df.drop('Unit', axis=1)

def prepare_pesticides_data(df):
    df = df.drop(['Domain', 'Element'], axis=1)
    df.rename({'Area': 'Country', 'Value': 'Pesticides (tonnes)'}, axis=1, inplace=True)
    return df.drop(['Unit', 'Item'], axis=1)

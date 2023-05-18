# data_merging.py
import pandas as pd

def merge_dataframes(rainfall_df, temperature_df, yield_df, pesticides_df):
    rainfall_df.rename({'Rainfall - (MM)': 'Rainfall (mm)'}, axis=1, inplace=True)
    temperature_df.rename({'Temperature - (Celsius)': 'Temperature (Celsius)'}, axis=1, inplace=True)
    yield_df.rename({'Area': 'Country', 'Value': 'Yield (hg/ha)'}, axis=1, inplace=True)
    pesticides_df.rename({'Area': 'Country', 'Value': 'Pesticides (tonnes)'}, axis=1, inplace=True)
    rain_temp_df = pd.merge(rainfall_df, temperature_df, on=['Country', 'Year'])
    rain_temp_yield_df = pd.merge(rain_temp_df, yield_df, on=['Country', 'Year'])
    rain_temp_yield_pest_df = pd.merge(rain_temp_yield_df, pesticides_df, on=['Country', 'Year'])
    rain_temp_yield_pest_df.drop(['ISO3_x','ISO3_y'], axis=1, inplace=True)
    data = rain_temp_yield_pest_df[['Year', 'Country', 'Item', 'Rainfall (mm)', 'Temperature (Celsius)', 'Pesticides (tonnes)', 'Yield (hg/ha)']]
    return data


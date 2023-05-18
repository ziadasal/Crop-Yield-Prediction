import tkinter as tk
from tkinter import messagebox
import data_loading
import data_preparation
import data_merging
import data_preprocessing
import data_modeling
import data_modelingAll
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import graph 


class Crop_analysis:
    def __init__(self, rainfall_data, temperature_data, yield_data, pesticides_data, models):
        self.models = models

        # Create initial dataframe
        self.data = data_merging.merge_dataframes(rainfall_data, temperature_data, yield_data, pesticides_data)

        # Get unique countries and crops
        self.countries = self.data['Country'].unique().tolist()
        self.crops = self.data['Item'].unique().tolist()

        # Create GUI
        self.window = tk.Tk()
        self.window.title("Crop Recommendation System")
        self.window.geometry("500x500")

        # Create a menu bar
        self.menu_bar = tk.Menu(self.window)
        
        # Create a Help menu
        self.help_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.help_menu.add_command(label="R-squared (R^2)", command=self.show_r_squared)
        self.help_menu.add_command(label="Mean Absolute Error (MAE)", command=self.show_mae)
        self.help_menu.add_command(label="Mean Squared Error (MSE)", command=self.show_mse)
        self.help_menu.add_command(label="Root Mean Squared Error (RMSE)", command=self.show_rmse)
        self.help_menu.add_command(label="Maximum Error", command=self.show_max_error)
        self.help_menu.add_command(label="Mean Absolute Percentage Error (MAPE)", command=self.show_mape)
        self.menu_bar.add_cascade(label="Help", menu=self.help_menu)

        # Configure the window to use the menu bar
        self.window.config(menu=self.menu_bar)

        # Model label and dropdown menu
        self.model_label = tk.Label(self.window, text="Select a model:")
        self.model_label.pack()
        self.model_var = tk.StringVar(self.window)
        self.model_var.set(self.models[0])  # Default option is the first model in the list
        self.model_dropdown = tk.OptionMenu(self.window, self.model_var, *self.models)
        self.model_dropdown.pack(pady=10)

        # Create a frame for the buttons
        self.buttons_frame1 = tk.Frame(self.window)
        self.buttons_frame1.pack(pady=20)

        # Model using selected button
        self.model_selected_button = tk.Button(self.buttons_frame1, text="Model using selected", command=self.model_using_selected)
        self.model_selected_button.pack(side="left",padx=10)

        # Model using all button
        self.model_all_button = tk.Button(self.buttons_frame1, text="Model using all", command=self.model_using_all)
        self.model_all_button.pack(side="left",padx=10)

        # Compare between all button
        self.compare_button = tk.Button(self.buttons_frame1, text="Compare between all", command=self.compare_between_all)
        self.compare_button.pack(side="left",padx=10)

        # Country label and dropdown menu
        self.country_label = tk.Label(self.window, text="Select a country:")
        self.country_label.pack()
        self.country_var = tk.StringVar(self.window)
        self.country_var.set("All")  # Default option is "All"
        self.country_dropdown = tk.OptionMenu(self.window, self.country_var, "All", *self.countries, command=self.update_countries)
        self.country_dropdown.pack(pady=10)

        # Create a frame for the visualization buttons
        self.vis_buttons_frame2 = tk.Frame(self.window)
        self.vis_buttons_frame2.pack(pady=20)


        # Temperature Count
        self.freq_button = tk.Button(self.vis_buttons_frame2, text="Temperature Count", command=self.temperature_count)
        self.freq_button.pack(side="left",padx=10)

        # Rainfall Count
        self.freq_button = tk.Button(self.vis_buttons_frame2, text="Rainfall Count", command=self.rainfall_count)
        self.freq_button.pack(side="left",padx=10)

        # Pestiside Count
        self.freq_button = tk.Button(self.vis_buttons_frame2, text="Pesticide Count", command=self.preticide_count)
        self.freq_button.pack(side="left",padx=10)

        # Crop label and dropdown menu
        self.crop_label = tk.Label(self.window, text="Select a crop:")
        self.crop_label.pack()
        self.crop_var = tk.StringVar(self.window)
        self.crop_var.set("All")  # Default option is "All"
        self.crop_dropdown = tk.OptionMenu(self.window, self.crop_var, "All", *self.crops, command=self.update_crops)
        self.crop_dropdown.pack(pady=10)

        # Create a frame for the visualization buttons
        self.vis_buttons_frame3 = tk.Frame(self.window)
        self.vis_buttons_frame3.pack(pady=20)

        # Yield Count
        self.freq_button = tk.Button(self.vis_buttons_frame3, text="Yield Count", command=self.yield_count)
        self.freq_button.pack(side='left',padx=10)

        # Create a frame for the visualization buttons
        self.vis_buttons_frame4 = tk.Frame(self.window)
        self.vis_buttons_frame4.pack(pady=20)

        # Mean harvested value button
        self.mean_button = tk.Button(self.vis_buttons_frame4, text="Mean harvested value", command=self.plot_mean_harvested_value)
        self.mean_button.pack(side="left",padx=10)

        # Frequency Merged data Count button
        self.freq_button = tk.Button(self.vis_buttons_frame4, text="Frequency Merged data Count", command=self.frequency_count)
        self.freq_button.pack(side="left",padx=10)

        # graph button
        self.graph_button = tk.Button(self.vis_buttons_frame4, text="Graph", command=self.descition_tree_map)
        self.graph_button.pack(side="left",padx=10)


        # Result label
        self.result_label = tk.Label(self.window, text="")
        self.result_label.pack()

        # Run the GUI
        self.window.mainloop()
    
    # Add functions to show explanations for each metric
    def show_r_squared(self):
        messagebox.showinfo("R-squared (R^2)", "R-squared (R^2) measures how well the regression line fits the data. R-squared ranges from 0 to 1, with 1 indicating a perfect fit. A higher R-squared value generally indicates that the model fits the data better.")

    def show_mae(self):
        messagebox.showinfo("Mean Absolute Error (MAE)", "Mean Absolute Error (MAE) measures the average absolute difference between the predicted and actual values. MAE ranges from 0 to infinity, with 0 indicating a perfect fit. A lower MAE value generally indicates that the model is more accurate.")

    def show_mse(self):
        messagebox.showinfo("Mean Squared Error (MSE)", "Mean Squared Error (MSE) measures the average squared difference between the predicted and actual values. MSE ranges from 0 to infinity, with 0 indicating a perfect fit. A lower MSE value generally indicates that the model is more accurate.")

    def show_rmse(self):
        messagebox.showinfo("Root Mean Squared Error (RMSE)", "Root Mean Squared Error (RMSE) measures the square root of the average squared difference between the predicted and actual values. RMSE ranges from 0 to infinity, with 0 indicating a perfect fit. A lower RMSE value generally indicates that the model is more accurate.")

    def show_max_error(self):
        messagebox.showinfo("Maximum Error", "Maximum Error measures the largest absolute difference between the predicted and actual values. Maximum error measures the largest absolute difference between the predicted and actual values. A lower maximum error generally indicates that the model is more accurate. Maximum error measures the largest absolute difference between the predicted and actual values. A lower maximum error generally indicates that the model is more accurate.")

    def show_mape(self):
        messagebox.showinfo("Mean Absolute Percentage Error (MAPE)", "Mean Absolute Percentage Error (MAPE) measures the average percentage difference between the predicted and actual values. MAPE measures the average percentage difference between the predicted and actual values. A lower MAPE value generally indicates that the model is more accurate. However, MAPE should be used with caution as it can be misleading when actual values are close to zero.")
    
    def update_countries(self, value):
        # Update countries variable with current values from dataframe
        self.countries = self.data['Country'].unique().tolist()

    def update_crops(self, value):
        # Update crops variable with current values from dataframe
        self.crops = self.data['Item'].unique().tolist()

    def update_country(self, *args):
            self.country = self.country_var.get()

    def update_crop(self, *args):
        self.crop = self.crop_var.get()
    
    def yield_count(self):
        country = self.country_var.get()
        crop = self.crop_var.get()
        if country != 'All' and crop != 'All':
           yield_data.loc[(yield_data['Country'] == country) & (yield_data['Item']== crop)].groupby('Year').mean(numeric_only=True).plot()
           plt.title('Yield Count for ' + crop + ' in ' + country)
           plt.ylabel('Yield (hg/ha)')
           plt.xlabel('Year')
        elif country != 'All':
            yield_data.loc[yield_data['Country'] == country].groupby('Year').mean(numeric_only=True).plot()
            plt.title('Yield Count for ' + country)
            plt.ylabel('Yield (hg/ha)')
            plt.xlabel('Year')
        elif crop != 'All':
            yield_data.loc[yield_data['Item']== crop].groupby('Year').mean(numeric_only=True).plot()
            plt.title('Yield Count for ' + crop)
            plt.ylabel('Yield (hg/ha)')
            plt.xlabel('Year')
        else:
            # prepare data
            yield_mean = yield_data.groupby(['Year', 'Item']).mean(numeric_only=True)
            # plot data
            print(yield_mean)
            fig, ax = plt.subplots(figsize=(15,9))
            fig.suptitle('Mean harvested value across all countries between 1961 and 2019')
            yield_mean['Yield (hg/ha)'].unstack().plot(ax=ax)
            ax.set_ylabel('Mean Value across all countries')
            ax.set_xlabel('Year')

        plt.show()
        
    def temperature_count(self):
        country = self.country_var.get()
        if country != 'All':
            temperature_data.loc[temperature_data['Country'] == country].groupby('Year').mean(numeric_only=True).plot()
            plt.title('Temperature in ' + country)
            plt.xlabel('Year')
            plt.ylabel('Temperature (C)')
        else:
            temperature_data.groupby('Year').mean(numeric_only=True).plot()
            plt.title('Temperature across all countries')
            plt.xlabel('Year')
            plt.ylabel('Temperature (C)')

        plt.show()  

    def rainfall_count(self):
        country = self.country_var.get()
        if country != 'All':
            rainfall_data.loc[rainfall_data['Country'] == country].groupby('Year').mean(numeric_only=True).plot()
            plt.title('Rainfall in ' + country)
            plt.xlabel('Year')
            plt.ylabel('Rainfall (mm)')
        else:
            rainfall_data.groupby('Year').mean(numeric_only=True).plot()
            plt.title('Rainfall across all countries')
            plt.xlabel('Year')
            plt.ylabel('Rainfall (mm)')

        plt.show()

    def preticide_count(self):
        country = self.country_var.get()
        if country != 'All':
            pesticides_data.loc[pesticides_data['Country'] == country].groupby('Year').mean(numeric_only=True).plot()
            plt.title('Pesticides used in ' + country)
            plt.xlabel('Year')
            plt.ylabel('Pesticides used (tonnes)')
        else:
            pesticides_data.groupby('Year').mean(numeric_only=True).plot()
            plt.title('Pesticides used across all countries')
            plt.xlabel('Year')
            plt.ylabel('Pesticides used (tonnes)')
        plt.show()


    def frequency_count(self):
        country = self.country_var.get()
        crop = self.crop_var.get()
        df=self.data
        if country != 'All' and crop != 'All':
            df=self.data[(self.data['Country']==country) & (self.data['Item']==crop)]
        elif country != 'All':
            df=self.data[(self.data['Country']==country)]
        elif crop != 'All':
            df=self.data[(self.data['Item']==crop)]

        fig, axs = plt.subplots(1,1, figsize=(9,9))
        fig.suptitle('Frequency Count', size=30)
        temp_df = df['Item'].value_counts().to_frame().reset_index()
        g = sns.barplot(x='Item', y='count', data=temp_df, orient='v', ax=axs)
        for index, row in temp_df.iterrows():
            g.text(row.name,row['count']+50, row['count'], color='black', ha="center")
        axs.set_xlabel('Item')
        axs.set_ylabel('Count')
        plt.xticks(rotation=90)
        plt.show()

    def plot_mean_harvested_value(self):
        country = self.country_var.get()
        crop = self.crop_var.get()
        df=self.data
        if country != 'All' and crop != 'All':
            df=self.data[(self.data['Country']==country) & (self.data['Item']==crop)]
        elif country != 'All':
            df=self.data[(self.data['Country']==country)]
        elif crop != 'All':
            df=self.data[(self.data['Item']==crop)]

        temp_data = df.groupby(['Year', 'Item']).mean(numeric_only=True)
        fig, ax = plt.subplots(figsize=(9,9))

        if(country != 'All' and crop != 'All'):
            fig.suptitle('Mean harvested value for '+country+' for '+crop+' between 1961 and 2019')
        elif(country != 'All'):
            fig.suptitle('Mean harvested value for '+country+' between 1961 and 2019')
        elif(crop != 'All'):
            fig.suptitle('Mean harvested value for '+crop+' between 1961 and 2019')
        else:
            fig.suptitle('Mean harvested value for all countries between 1961 and 2019')
        
        temp_data['Yield (hg/ha)'].unstack().plot(ax=ax)
        if country != 'All' and crop != 'All':
            ax.set_ylabel('Mean Value for '+country+' for '+crop)
        elif country != 'All':
            ax.set_ylabel('Mean Value for '+country)
        elif crop != 'All':
            ax.set_ylabel('Mean Value for '+crop)
        else:
            ax.set_ylabel('Mean Value')
        ax.set_xlabel('Year')
        plt.show()
        

    def model_using_selected(self):
        model = self.model_var.get()
        # Call function to get recommended crop based on selected options
        df=data_preprocessing.one_hot_encode(self.data)
        df=data_preprocessing.feature_scaling(df)
        df=data_preprocessing.filter_data(df) 
        return data_modeling.run_regression(df, model)
    
    def model_using_all(self):
        # Call function to get recommended crop based on selected options
        df=data_preprocessing.one_hot_encode(self.data)
        df=data_preprocessing.feature_scaling(df)
        df=data_preprocessing.filter_data(df) 
        return data_modelingAll.regression_analysis(df)
    def compare_between_all(self):
        # Call function to get recommended crop based on selected options
        df=data_preprocessing.one_hot_encode(self.data)
        df=data_preprocessing.feature_scaling(df)
        df=data_preprocessing.filter_data(df) 
        return data_modelingAll.evaluate_estimators(df)
    
    def descition_tree_map(self):
        country = self.country_var.get()
        crop = self.crop_var.get()
        df=self.data
        if country != 'All' and crop != 'All':
            df=self.data[(self.data['Country']==country) & (self.data['Item']==crop)]
        elif country != 'All':
            df=self.data[(self.data['Country']==country)]
        elif crop != 'All':
            df=self.data[(self.data['Item']==crop)]
        df=data_preprocessing.one_hot_encode(self.data)
        df=data_preprocessing.feature_scaling(df)
        df=data_preprocessing.filter_data(df) 
        return graph.visualize_tree(data=df,feature_names=df.columns[:-1])
    
    

pesticides_data=data_loading.load_pesticides_data()
rainfall_data=data_loading.load_rainfall_data()
temperature_data=data_loading.load_temperature_data()
yield_data=data_loading.load_yield_data()

pesticides_data=data_preparation.prepare_pesticides_data(pesticides_data)
rainfall_data=data_preparation.prepare_rainfall_data(rainfall_data)
temperature_data=data_preparation.prepare_temperature_data(temperature_data)
yield_data=data_preparation.prepare_yield_data(yield_data)

models = ['Linear Regression',
          'Decision Tree Regression',
          'Stochastic Gradient Descent Regression',
          'Gradient Boosting Regression',
          'Random Forest Regression',
          'K-nearest Neighbour 5',]

CropAnalysis = Crop_analysis(rainfall_data,temperature_data,yield_data,pesticides_data,models)


import tkinter as tk
import data_loading
import data_preparation
import data_merging
import data_preprocessing
import data_modeling
import data_prediction

class CropRecommendation:
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
        self.window.geometry("400x450")

        # Model label and dropdown menu
        self.model_label = tk.Label(self.window, text="Select a model:")
        self.model_label.pack()
        self.model_var = tk.StringVar(self.window)
        self.model_var.set(self.models[1])  # Default option is the secound model in the list
        self.model_dropdown = tk.OptionMenu(self.window, self.model_var, *self.models)
        self.model_dropdown.pack(pady=10)

        # Country label and dropdown menu
        self.country_label = tk.Label(self.window, text="Select a country:")
        self.country_label.pack()
        self.country_var = tk.StringVar(self.window)
        self.country_var.set("All")  # Default option is "All"
        self.country_dropdown = tk.OptionMenu(self.window, self.country_var, "All", *self.countries, command=self.update_countries)
        self.country_dropdown.pack(pady=10)

        # Crop label and dropdown menu
        self.crop_label = tk.Label(self.window, text="Select a crop:")
        self.crop_label.pack()
        self.crop_var = tk.StringVar(self.window)
        self.crop_var.set("All")  # Default option is "All"
        self.crop_dropdown = tk.OptionMenu(self.window, self.crop_var, "All", *self.crops, command=self.update_crops)
        self.crop_dropdown.pack(pady=10)

        # Rainfall label and entry
        self.year_label = tk.Label(self.window, text="Year :")
        self.year_label.pack()
        self.year_entry = tk.Entry(self.window)
        self.year_entry.pack()


        # Rainfall label and entry
        self.rainfall_label = tk.Label(self.window, text="Rainfall (mm): (Normal is between 500 to 1700 mm)")
        self.rainfall_label.pack()
        self.rainfall_entry = tk.Entry(self.window)
        self.rainfall_entry.pack()

        # Temperature label and entry
        self.temperature_label = tk.Label(self.window, text="Temperature (Celsius): (Normal is between 10 to 30 C)")
        self.temperature_label.pack()
        self.temperature_entry = tk.Entry(self.window)
        self.temperature_entry.pack()

        # Pesticides label and entry
        self.pesticides_label = tk.Label(self.window, text="Pesticides (tonnes): (Normal is between 99 to 7420 tonnes)")
        self.pesticides_label.pack()
        self.pesticides_entry = tk.Entry(self.window)
        self.pesticides_entry.pack()

        # Create a frame for the buttons
        self.buttons_frame1 = tk.Frame(self.window)
        self.buttons_frame1.pack(pady=20)

        # Model using selected button
        self.model_selected_button = tk.Button(self.buttons_frame1, text="Model using selected", command=self.model_using_selected)
        self.model_selected_button.pack(side="left",padx=10)
        # Model using all button
        self.model_all_button = tk.Button(self.buttons_frame1, text="Predict", command=self.predict)
        self.model_all_button.pack(side="left",padx=10)

        # Result label
        self.result_label = tk.Label(self.window, text="")
        self.result_label.pack()

        # Run the GUI
        self.window.mainloop()
    
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
     
    def model_using_selected(self):
        model = self.model_var.get()
        # Call function to get recommended crop based on selected options
        df=data_preprocessing.one_hot_encode(self.data)
        df=data_preprocessing.feature_scaling(df)
        df=data_preprocessing.filter_data(df) 
        return data_modeling.run_regression(df, model)
    
    def predict (self):
        model = self.model_var.get()
        year = int(self.year_entry.get())
        rainfall = float(self.rainfall_entry.get())
        temperature = float(self.temperature_entry.get())
        pesticides = float(self.pesticides_entry.get())
        country = self.country_var.get()
        crop = self.crop_var.get()
        return data_prediction.predict(self.data, model, year, rainfall, temperature, pesticides, country, crop)

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

recommendation_system = CropRecommendation(rainfall_data,temperature_data,yield_data,pesticides_data,models)


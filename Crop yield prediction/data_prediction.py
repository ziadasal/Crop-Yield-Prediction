import numpy as np
import pandas as pd
import data_preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from PIL import Image

def predict(data, model, year, rainfall, temperature, pesticides, country, crop):
    df = data_preprocessing.one_hot_encode(data)
    df = data_preprocessing.feature_scaling(df)
    df = data_preprocessing.filter_data(df)

    y = df['Yield (hg/ha)']
    X = df.drop('Yield (hg/ha)', axis=1)

    if (model == 'Linear Regression'):
        model = LinearRegression()
    elif (model == 'Decision Tree Regression'):
        model = DecisionTreeRegressor()
    elif (model == 'Stochastic Gradient Descent Regression'):
        model = SGDRegressor()
    elif (model == 'Gradient Boosting Regression'):
        model = GradientBoostingRegressor()
    elif (model == 'Random Forest Regression'):
        model = RandomForestRegressor()
    elif (model == 'K-nearest Neighbour 5'):
        model = KNeighborsRegressor(n_neighbors=5)
    else:
        print('Error: Model not found')
        return
    
    crops = ['Barley', 'Maize', 'Potatoes', 'Wheat', 'Carrots and turnips','Cauliflowers and broccoli', 'Garlic', 'Hops', 'Oats','Onions, shallots, green', 'Rye', 'Sweet potatoes', 'Buckwheat', 'Ginger']
    tips = {
        'Barley': 'Use nitrogen fertilizer to improve yield.',
        'Maize': 'Plant in well-drained soil and use irrigation to improve yield.',
        'Potatoes': 'Control pests and disease using appropriate measures to improve yield.',
        'Wheat': 'Apply nitrogen fertilizer at the right time to improve yield.',
        'Carrots and turnips': 'Ensure proper soil moisture and use appropriate pest management techniques to improve yield.',
        'Cauliflowers and broccoli': 'Plant in cool, moist conditions to improve yield.',
        'Garlic': 'Plant in well-drained soil and use adequate amounts of nitrogen and phosphorus to improve yield.',
        'Hops': 'Use appropriate trellising and pest management techniques to improve yield.',
        'Oats': 'Apply nitrogen fertilizer at the right time and control weeds to improve yield.',
        'Onions, shallots, green': 'Plant in well-drained soil and use adequate amounts of nitrogen to improve yield.',
        'Rye': 'Plant in well-drained soil and use appropriate amounts of fertilizer to improve yield.',
        'Sweet potatoes': 'Ensure proper soil moisture and use appropriate pest management techniques to improve yield.',
        'Buckwheat': 'Plant in well-drained soil and use appropriate amounts of fertilizer to improve yield.',
        'Ginger': 'Plant in warm, moist conditions and use appropriate amounts of fertilizer to improve yield.'
    }
    if (crop == 'All'):
        new_data = {
            'Year': [year]*len(crops),
            'Country': [country]*len(crops),
            'Item': crops,
            'Rainfall (mm)': [rainfall]*len(crops),
            'Temperature (Celsius)': [temperature]*len(crops),
            'Pesticides (tonnes)': [pesticides]*len(crops)
        }
    else:   
        new_data = {
            'Year': [year],
            'Country': [country],
            'Item': [crop],
            'Rainfall (mm)': [rainfall],
            'Temperature (Celsius)': [temperature],
            'Pesticides (tonnes)': [pesticides]
        }

    new_data_df = pd.DataFrame(new_data)
    new_data_processed = pd.get_dummies(new_data_df, columns=['Country', 'Item'], prefix=['Country', 'Item'])
    new_data_processed = new_data_processed.reindex(columns=df.columns, fill_value=0)  # Match columns with the trained data

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    y_new = model.predict(new_data_processed.drop('Yield (hg/ha)', axis=1))

    # Load and display the crop image
    if (crop != 'All'):
        crop_image = Image.open('./crops/{crop}.jpg'.format(crop=crop))
        plt.imshow(crop_image)
        plt.text(0, 500, f'Predicted Yield {crop} for {year} and with Rainfall {rainfall} mm and Temp{temperature}°C: {int(y_new[0])}', color='white', backgroundcolor='black')
        plt.text(0, 525, tips[crop], color='green', backgroundcolor='black')
        average_yield = np.mean(y)
        if y_new < average_yield:
            plt.text(0, 0, 'Yield below average. Not recommended.', color='red', backgroundcolor='black')

        plt.axis('off')
        # Show the plot
        plt.show()
    else:
        sorted_crops_yields = sorted(zip(crops, y_new), key=lambda x: x[1], reverse=True)
        sorted_crops = [crop for crop, _ in sorted_crops_yields]
        sorted_yields = [yield_ for _, yield_ in sorted_crops_yields]

        fig, (crop_ax, bar_ax) = plt.subplots(1, 2, figsize=(10, 5))

        # Display the crop image and text on the right
        crop_image = Image.open('./crops/{crop}.jpg'.format(crop=sorted_crops[0]))
        crop_ax.imshow(crop_image)
        crop_ax.text(0, 500, f'Predicted Yield for{sorted_crops[0]} is ', color='white', backgroundcolor='black')
        crop_ax.text(0, 525, f'for {year} and with Rainfall {rainfall}mm and Temp {temperature}°C: {int(sorted_yields[0])}', color='white', backgroundcolor='black')
        crop_ax.text(0, 560, tips[sorted_crops[0]], color='green', backgroundcolor='black')
        crop_ax.axis('off')

        # Display the bar plot on the left
        bar_ax.bar(sorted_crops, sorted_yields)
        bar_ax.set_xticklabels(sorted_crops, rotation=90)
        bar_ax.set_xlabel('Crop')
        bar_ax.set_ylabel('Yield (hg/ha)')
        bar_ax.set_title('Predicted Yield for {year}and with Rainfall {rainfall}mm and Temp {temperature}°C'.format(year=year, rainfall=rainfall, temperature=temperature))

        # Adjust the spacing between subplots
        fig.subplots_adjust(wspace=0.3)

        # Show the plot
        plt.show()
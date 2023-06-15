# Crop Yield Prediction

<p align="center">
  <img src="https://github.com/ziadasal/Crop-Yield-Prediction/blob/main/Images/GUI%20of%20version%202.png" alt="GUIVersion 2">
</p>

This repository contains the code and datasets for the Crop Yield Prediction project. The objective of this project is to develop a predictive model that can accurately forecast crop yields based on various factors such as weather conditions, soil properties, and crop management practices.

## Datasets

The datasets used for this project are stored in the `datasets` folder. These datasets contain historical data on weather patterns, soil characteristics, and crop yield measurements. The dataset files are in CSV format and are named accordingly to indicate their contents.

## Code Files

The repository includes several code files that perform various tasks related to data loading, data preparation, modeling, and prediction. Here is a brief description of each file:

- `data_loading.py`: This file is responsible for loading the dataset files and combining them into a single dataset for analysis and modeling.
- `data_merging.py`: This file contains the code to merge the different datasets based on common identifiers or keys.
- `data_preparation.py`: This file handles the preprocessing and preparation of the data, including handling missing values, scaling, and feature engineering.
- `data_modeling.py`: This file focuses on building machine learning models for crop yield prediction. It includes the implementation of various algorithms and techniques, such as regression models, decision trees, random forests, or neural networks.
- `data_prediction.py`: This file utilizes the trained models to make crop yield predictions on new or unseen data.
- `graph.py`: This file contains code for generating visualizations and graphs to analyze the data and communicate the results.

Additionally, there are two GUI versions (`GuiVersion1.py` and `GuiVersion2.py`) that provide a graphical user interface for interacting with the models and obtaining predictions.

## Usage

To run the Crop Yield Prediction project, follow these steps:

1. Ensure that the necessary datasets are stored in the `datasets` folder.
2. Run the `data_loading.py` file to load and merge the datasets.
3. Execute the `data_preparation.py` file to preprocess and prepare the data.
4. Run the `data_modeling.py` file to train the crop yield prediction models.
5. Utilize the trained models for predictions by running the `data_prediction.py` file.
6. Optionally, use the GUI versions (`GuiVersion1.py` or `GuiVersion2.py`) for a more interactive experience.

Make sure to customize the code and parameters according to your specific requirements and dataset characteristics.

## From the program
![GUIVersion 1](https://github.com/ziadasal/Crop-Yield-Prediction/blob/main/Images/Gui%20of%20version%201.png)
![The count of Each Crop Record After merging data](https://github.com/ziadasal/Crop-Yield-Prediction/blob/main/Images/The%20count%20of%20Each%20Crop%20Record%20After%20merging%20data.png)
![Using all models](https://github.com/ziadasal/Crop-Yield-Prediction/blob/main/Images/Using%20all%20models.png)
![Recommendation for Egypt with some inputs](https://github.com/ziadasal/Crop-Yield-Prediction/blob/main/Images/Recommendation%20for%20Egypt%20with%20some%20inputs.png)
![not recommended in these conditions](https://github.com/ziadasal/Crop-Yield-Prediction/blob/main/Images/Barley%20is%20not%20recommended%20in%20these%20conditions.png)
## Conclusion

The Crop Yield Prediction project provides a framework for accurately forecasting crop yields based on various factors. By leveraging historical data and utilizing machine learning techniques, this project can assist farmers, researchers, and agricultural stakeholders in making informed decisions related to crop planning, resource allocation, and yield optimization.

For more details and code implementation, please refer to the respective files in this repository.

## References

No external references are mentioned in the provided files.

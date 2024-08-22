# EMG-Muscle-classification

This project focuses on the classification of muscle activity using Electromyography (EMG) data. The dataset comprises EMG signals collected from 24 participants during various muscle activities. The primary goal of this project is to build and train machine learning models to accurately classify these muscle activities based on the EMG data.

## Project Overview

- **Data Collection**: The EMG data was collected from 24 participants performing different muscle activities. The dataset is provided in a compressed file (`data.rar`), which contains the raw data files.
- **Data Transformation**: The raw data needs to be transformed into a structured CSV format for further analysis and model training.
- **Model Training**: Once the data is in CSV format, various machine learning models can be trained using the provided Jupyter Notebook.

## How to Run the Project

Follow these steps to set up and run the project:

### 1. Install the Required Dependencies

1. Ensure you have Python 3.x installed. (We have used Python 3.12.4 version)
2. Install the required Python packages by running the following command:

   ```bash
   pip install -r requirements.txt
   ```

   This command will install all the necessary dependencies listed in the `requirements.txt` file.

### 2. Extract the Data

1. Download the `data.rar` file.
2. Extract the `data.rar` file in the same directory where the project files are located. This will create a `data/` folder containing the raw data files.

### 3. Transform the Data

1. Run the `transfordata.py` script to transform the raw data into CSV format. This script processes the raw EMG data files and converts them into a CSV file that can be used for model training.
   
   ```bash
   python transfordata.py
   ```

2. After running the script, you should have a CSV file containing the transformed data.

### 4. Train the Model

1. Open the provided Jupyter Notebook (`train_model.ipynb`) in your preferred Jupyter environment.
2. The notebook contains the code to load the CSV data, preprocess it, and train machine learning models to classify muscle activity based on the EMG signals.
3. Run the cells in the notebook sequentially to train and evaluate the models.

## Conclusion

you can easily run the project and experiment with different models and techniques to classify muscle activities based on EMG data.
```

# GIBots-ML-Assignment

# Data Preprocessing and Prediction Tool

This repository contains a Python script that allows you to load, preprocess, and make predictions on your data using machine learning techniques. The script also provides a graphical user interface (GUI) to facilitate the process.

## Prerequisites

Before running this code, ensure that you have installed the following libraries:

- tkinter (comes pre-installed with Python)
- numpy
- pandas
- scikit-learn
- skmultilearn
- gensim
- matplotlib

You can install these libraries by running the following command:

## Running the Code

1. Execute the Python file in your terminal or command prompt, or open it in an Integrated Development Environment (IDE) like PyCharm, Visual Studio Code, or Spyder.

2. A window titled "Data Preprocessing and Prediction" will appear, containing three buttons and a text area.

## Using the GUI

### Load Data and Preprocess

Click this button to load and preprocess your data files. You will be prompted to select three CSV files: Train Data File, Test Data File, and Train Labels File. After selecting the files, the code will preprocess the data, and a success message will be displayed in the text area.

### Save Submission Data

After preprocessing the data, click this button to save the submission data to a CSV file. You will be prompted to choose a location and filename for the CSV file. Once the file is saved, a success message will be displayed in the text area.

### Visualize Word Embeddings

This button will visualize the word embeddings from the 'x3' and 'x4' columns of your train data. A new window will open, displaying a scatter plot of the word embeddings. If the 'x3' or 'x4' columns are not found in the train data, an error message will be displayed in the text area.

## Interpreting the Results

After running the code and saving the submission data, open the generated CSV file to view the results. The CSV file will contain two columns: 'id' and the predicted labels for each instance in the test data.

## Error Handling

If you encounter any errors during the execution of the code, the error messages will be displayed in the text area of the GUI window.

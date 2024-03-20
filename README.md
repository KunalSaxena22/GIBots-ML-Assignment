# GIBots-ML-Assignment

# Getting Started

## Introduction of ML Text Classification

This repository contains code for developing a machine learning model to predict the probability of text belonging to specific classes. The project utilizes techniques like Bag of Words, TF-IDF vectorization, and word embeddings, and explores the use of the Hash field value.

## Data
- `train.csv`: Features for the training set
- `trainLabels.csv`: Labels for the training set
- `test.csv`: Features for the test set
- `sampleSubmission.csv`: Example submission format

## Features
- Content: Text hash
- Parsing: N-gram type (number, text, alphanumeric)
- Spatial: N-gram position and size
- Relational: Nearby text details

## Labels
Multi-label classification problem where a sample can belong to multiple classes.

## Objective
Develop a machine learning model to accurately predict the probabilities of text samples belonging to different classes using the provided features and labels.


# How to Use the Repository

## Step 1: Clone the Repository

First, you need to clone the GitHub repository containing the code to your local machine. Open your terminal or command prompt and navigate to the directory where you want to clone the repository. Then, run the following command:

```bash
https://github.com/KunalSaxena22/GIBots-ML-Assignment.git
```

## Step 2: Install Required Libraries

After cloning the repository, navigate to the cloned directory:

```bash
cd GIBots-ML-Assignment
```

Before running the code, you need to ensure that you have installed the required libraries. Open the `requirements.txt` file in the repository and note down the libraries listed there.

You can install these libraries by running the following command in your terminal or command prompt:

```bash
pip install -r requirements.txt
```

This command will install all the required libraries listed in the `requirements.txt` file.

## Step 3: Run the Code

Once you have installed the required libraries, you can run the code by executing the Python file in your terminal or command prompt:

```bash
python main.py
```

Replace `main.py` with the actual name of the Python file containing the code.

## Step 4: Follow the GUI Instructions

After running the code, a window titled "Data Preprocessing and Prediction" will appear. Follow the same instructions provided in the previous documentation for using the Graphical User Interface (GUI):

- Click the "Load and Process Data" button to load and preprocess your data files.
- Click the "Save Submission Data" button to save the submission data to a CSV file.
- Click the "Exit" button to close the application.

```

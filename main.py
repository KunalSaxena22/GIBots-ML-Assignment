#Kunal Code
import tkinter as tk
from tkinter import filedialog
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Function to load and preprocess data
def load_and_preprocess_data():
    global X_train, X_test, Y_train, submission_df, train_data

    try:
        # Loading data from local files
        train_data_file = filedialog.askopenfilename(title="Select Train Data File", filetypes=[("CSV Files", "*.csv")])
        train_df = pd.read_csv(train_data_file)

        test_data_file = filedialog.askopenfilename(title="Select Test Data File", filetypes=[("CSV Files", "*.csv")])
        test_df = pd.read_csv(test_data_file)

        train_target_file = filedialog.askopenfilename(title="Select Train Labels File", filetypes=[("CSV Files", "*.csv")])
        train_target_df = pd.read_csv(train_target_file)

        # Renaming columns for test data
        test_df.columns = ['id'] + [f'x{i}' for i in range(1, 146)]
        train_data = train_df

        # Preprocess train data
        if 'x3' in train_df.columns and 'x4' in train_df.columns:
            train_df['x3'] = train_df['x3'].fillna('').astype(str)
            train_df['x4'] = train_df['x4'].fillna('').astype(str)
            corpus = train_df['x3'] + train_df['x4']
        else:
            raise ValueError("Error: 'x3' or 'x4' column not found in train data.")

        vectorizer = TfidfVectorizer()
        X_train = vectorizer.fit_transform(corpus)
        Y_train = train_target_df.values

        # Loading data
        train_df = train_df
        test_df = test_df
        train_target_df = train_target_df
        # submission_df = pd.read_csv('/content/sampleSubmission_small.csv')

        # Renaming columns for test data
        test_df.columns = ['id'] + [f'x{i}' for i in range(1, 146)]

        # Preprocessing the Data for ML
        missing_percent_training_data = train_df.isna().sum().sort_values(ascending=False) * 100 / len(train_df)
        missing_percent_train_target_data = train_target_df.isna().sum().sort_values(ascending=False) * 100 / len(
            train_target_df)
        missing_percent_test_data = test_df.isna().sum().sort_values(ascending=False) * 100 / len(test_df)

        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        numeric_columns = train_df.select_dtypes(include=numerics).columns.tolist()

        object_columns = train_df.select_dtypes(include=object).columns

        train_df[object_columns] = train_df[object_columns].astype(str)
        test_df[object_columns] = test_df[object_columns].astype(str)

        boolean_columns = []
        alphanumeric_columns = []

        for col in object_columns:
            if train_df[col][0].isupper():
                boolean_columns.append(col)
            else:
                alphanumeric_columns.append(col)

        train_df[boolean_columns] = train_df[boolean_columns].astype('category')
        test_df[boolean_columns] = test_df[boolean_columns].astype('category')

        from sklearn.impute import SimpleImputer

        num_imputer = SimpleImputer(strategy='mean')
        obj_imputer = SimpleImputer(strategy='most_frequent')

        num_imputer.fit(train_df[numeric_columns])
        train_df[numeric_columns] = num_imputer.transform(train_df[numeric_columns])

        num_imputer.fit(test_df[numeric_columns])
        test_df[numeric_columns] = num_imputer.transform(test_df[numeric_columns])
        obj_imputer.fit(train_df[object_columns])
        train_df[object_columns] = obj_imputer.transform(train_df[object_columns])

        obj_imputer.fit(test_df[object_columns])
        test_df[object_columns] = obj_imputer.transform(test_df[object_columns])

        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaler.fit(train_df[numeric_columns])

        train_df[numeric_columns] = scaler.transform(train_df[numeric_columns])
        test_df[numeric_columns] = scaler.transform(test_df[numeric_columns])

        X_train = train_df[numeric_columns + boolean_columns]
        X_test = test_df[numeric_columns + boolean_columns]
        Y_train = train_target_df.loc[:9998, 'y1':]

        # Display success message
        result_text.delete('1.0', tk.END)
        result_text.insert(tk.END, "Data preprocessing completed successfully.\n")

        # Example preprocessing for object columns using LabelEncoder
        label_encoder = LabelEncoder()
        for col in X_train.select_dtypes(include=['object']).columns:
            X_train.loc[:, col] = label_encoder.fit_transform(X_train[col].copy())  # Explicitly create a copy
            X_test.loc[:, col] = label_encoder.transform(X_test[col].copy())  # Explicitly create a copy

        # Now, your features should be in a suitable format for the classifier
        # Let's proceed with training the classifier

        # initialize binary relevance multi-label classifier
        # with a gaussian naive bayes base classifier
        classifier = BinaryRelevance(classifier=GaussianNB())

        # train
        classifier.fit(X_train.values, Y_train.values)

        # predict
        predictions = classifier.predict(X_test.values)

        prediction_arr = predictions.toarray()

        target = []
        for num in test_df['id']:
            for i in range(1, 34):
                target.append(str(num) + '_y' + str(i))

        submission_df = pd.DataFrame(target, columns=['id'])

        # Display the results
        result_text.delete('1.0', tk.END)
        result_text.insert(tk.END, submission_df.to_string())

    except Exception as e:
        result_text.delete('1.0', tk.END)
        result_text.insert(tk.END, f"Error: {str(e)}")

def visualize_word_embeddings():
    if 'x3' in train_data.columns and 'x4' in train_data.columns:
        sentences = [row['x3'] + ' ' + row['x4'] for _, row in train_data.iterrows()]
    else:
        result_text.delete('1.0', tk.END)
        result_text.insert(tk.END, "Error: 'x3' or 'x4' column not found in train data.\n")
        return

    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

    # Visualize word embeddings
    words = list(model.wv.index_to_key)
    word_vectors = model.wv[words]

    # Create a scatter plot
    fig, ax = plt.subplots()
    ax.scatter(word_vectors[:, 0], word_vectors[:, 1])
    for i, word in enumerate(words):
        ax.annotate(word, (word_vectors[i, 0], word_vectors[i, 1]))

    # Embed the plot in the GUI
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack()

# Function to save submission data to a CSV file
def save_submission_data():
    try:
        # Ask user to choose a location to save the CSV file
        save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])

        # Save submission_df to a CSV file
        submission_df.to_csv(save_path, index=False)

        print("File created successfully")
        # Display success message
        result_text.delete('1.0', tk.END)
        result_text.insert(tk.END, f"Submission data saved to {save_path}\n")
    except Exception as e:
        result_text.delete('1.0', tk.END)
        result_text.insert(tk.END, f"Error saving submission data: {str(e)}\n")

# Function to exit the application
def exit_application():
    root.destroy()

# Create the main window
root = tk.Tk()
root.title("Data Preprocessing and Prediction")

# Create the GUI elements
load_button = tk.Button(root, text="Load Data and Preprocess", command=load_and_preprocess_data)
load_button.pack(pady=20)

visualize_button = tk.Button(root, text="Visualize Word Embeddings", command=visualize_word_embeddings)
visualize_button.pack(pady=15)

save_button = tk.Button(root, text="Save Submission Data", command=save_submission_data)
save_button.pack(pady=10)

exit_button = tk.Button(root, text="Exit", command=exit_application)
exit_button.pack(pady=5)

result_label = tk.Label(root, text="Preprocessing Results:", font=("Helvetica", 12, "bold"))
result_label.pack(pady=10)

result_text = tk.Text(root, height=10, width=50)
result_text.pack()

# Run the main loop
root.mainloop()




# Function to exit the application
def exit_application():
    root.destroy()
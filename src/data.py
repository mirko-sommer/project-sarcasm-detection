import os
import json
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from tabulate import tabulate

nltk.download('punkt')

def load_data(path='project/data/Sarcasm_Headlines_Dataset_v2.json'):
    """
    Load the dataset from a JSON file.

    Args:
        path (str): The file path to the JSON dataset.

    Returns:
        pd.DataFrame: Loaded dataset as a pandas DataFrame.
    """
    data = pd.read_json(path, lines=True)  
    return data


def data_samples_and_stats(data):
    """
    Display a sample of the dataset and show statistics about label distribution.

    Args:
        data (pd.DataFrame): The dataset containing headlines and labels.
    """
    # Print samples table
    print(data.sample(n=10))

    data_counts = data["is_sarcastic"].value_counts()
    print(f"The datset contains {int(data_counts.sum())} data points.")
    
    sarcastic_count = data_counts[1]
    non_sarcastic_count = data_counts[0]

    # Create table of distribution of labels
    label_counts_table = pd.DataFrame({
        "Label": ["Sarcastic", "Non-Sarcastic"],
        "Count": [sarcastic_count, non_sarcastic_count]
    })
    print("\nDistribution of labels:")
    print(tabulate(label_counts_table, headers='keys', tablefmt='fancy_grid'))

    # Plot distribution of labels
    fig, ax = plt.subplots(figsize=(4,4))
    ax.pie([sarcastic_count, non_sarcastic_count], labels=["sarcastic", "non-sarcastic"], autopct='%1.1f%%')
    plt.title("Distribution data points")
    plt.savefig("project/results/plots/datapoint_distribution.png")
    plt.show()


def split_data(data):
    """
    Split the dataset into training and testing sets.

    Args:
        data (pd.DataFrame): The dataset containing headlines and labels.

    Returns:
        X_train, X_test, y_train, y_test:
            - X_train: The training features (headlines).
            - X_test: The test features (headlines).
            - y_train: The training labels (sarcastic or non-sarcastic).
            - y_test: The test labels (sarcastic or non-sarcastic).
    """
    X = data['headline']
    y = data['is_sarcastic']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test


def one_hot_encoding(top_tokens, X_train, X_test, stemming=False):
    """
    Perform one-hot encoding on the headlines using a fixed number of top tokens.

    Args:
        top_tokens (int): The number of top tokens (features) to keep.
        X_train (pd.Series): The training set headlines.
        X_test (pd.Series): The test set headlines.
        stemming (bool): Whether to stem the tokens before one hot encoding.

    Returns:
        X_train_one_hot, X_test_one_hot, feature_names:
            - X_train_one_hot: One-hot encoded training set.
            - X_test_one_hot: One-hot encoded test set.
            - feature_names: The names of the features (tokens) used in encoding.
    """
    # One Hot encoding of headlines
    if stemming is True:
        stemmer = PorterStemmer()
        analyzer = CountVectorizer(stop_words='english').build_analyzer()

        def stemmed_words(doc):
            return (stemmer.stem(w) for w in analyzer(doc))
        
        one_hot = CountVectorizer(binary=True, max_features=top_tokens, analyzer=stemmed_words)
    else:
        one_hot = CountVectorizer(binary=True, max_features=top_tokens, stop_words='english')

    X_train_one_hot = one_hot.fit_transform(X_train)  
    X_test_one_hot = one_hot.transform(X_test) # Use same vectorizer on test data
    feature_names = one_hot.get_feature_names_out()
    return X_train_one_hot, X_test_one_hot, feature_names


def save_data_with_one_hot(X_train, X_test, y_train, y_test, 
                           X_train_one_hot, X_test_one_hot, 
                           feature_names, train_data, test_data, output_dir, stemmer):
    """
    Save the training and test data, including one-hot encodings, to JSON files.

    Args:
        X_train (pd.Series): The training set headlines.
        X_test (pd.Series): The test set headlines.
        y_train (pd.Series): The training set labels.
        y_test (pd.Series): The test set labels.
        X_train_one_hot (csr_matrix): The one-hot encoded training data.
        X_test_one_hot (csr_matrix): The one-hot encoded test data.
        feature_names (list): List of feature names (tokens).
        train_data (pd.DataFrame): The original training data.
        test_data (pd.DataFrame): The original test data.
        output_dir (str): The directory to save the JSON files.
        stemmer (bool): If the one-hot encoded data has been stemmed.
    """
    train_one_hot = X_train_one_hot.toarray()
    test_one_hot = X_test_one_hot.toarray()

    train_records = [
        {
            "index": idx,
            "headline": X_train.iloc[idx],
            "is_sarcastic": int(y_train.iloc[idx]),
            "article_link": train_data.iloc[idx].get("article_link", None),
            "one_hot_encoding": {feature_names[i]: int(train_one_hot[idx][i]) for i in range(len(feature_names))}
        }
        for idx in range(len(X_train))
    ]

    test_records = [
        {
            "index": idx,
            "headline": X_test.iloc[idx],
            "is_sarcastic": int(y_test.iloc[idx]),
            "article_link": test_data.iloc[idx].get("article_link", None),
            "one_hot_encoding": {feature_names[i]: int(test_one_hot[idx][i]) for i in range(len(feature_names))}
        }
        for idx in range(len(X_test))
    ]

    os.makedirs(output_dir, exist_ok=True)

    if stemmer:
        train_path = os.path.join(output_dir, "stemmed", f"{len(feature_names)}_train_data.json")
        test_path = os.path.join(output_dir, "stemmed", f"{len(feature_names)}_test_data.json")
    else: 
        train_path = os.path.join(output_dir, "not_stemmed", f"{len(feature_names)}_train_data.json")
        test_path = os.path.join(output_dir, "not_stemmed", f"{len(feature_names)}_test_data.json")

    with open(train_path, "w") as f:
        for record in train_records:
            f.write(json.dumps(record) + "\n")

    with open(test_path, "w") as f:
        for record in test_records:
            f.write(json.dumps(record) + "\n")

    print(f"Train and test data saved to {train_path} and {test_path}")


def load_preprocessed_data(top_tokens, stemmed, vectorizer):
    """
    Loads and preprocesses the training and test data.

    Args:
        top_tokens (int): Number of top tokens (features) to use.
        stemmed (bool): Whether to use the stemmed dataset.
        vectorizer (DictVectorizer): The vectorizer to transform data.

    Returns:
        tuple: X_train, y_train, X_test, y_test, feature_names
    """
    stemmed_path = "stemmed" if stemmed else "not_stemmed"
    test_data = load_data(f"project/data/splits/{stemmed_path}/{top_tokens}_test_data.json")
    train_data = load_data(f"project/data/splits/{stemmed_path}/{top_tokens}_train_data.json")

    X_test = vectorizer.fit_transform(test_data["one_hot_encoding"])
    y_test = test_data["is_sarcastic"]

    X_train = vectorizer.fit_transform(train_data["one_hot_encoding"])
    y_train = train_data["is_sarcastic"]

    feature_names = vectorizer.get_feature_names_out()

    return X_train, y_train, X_test, y_test, feature_names



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sarcasm Headline Classification")
    # Allow top_tokens to be a list of integers
    parser.add_argument('--top_tokens', type=int, nargs='+', default=[100], 
                        help="List of the number of top tokens (features) to include for one-hot encoding.")
    parser.add_argument('--stemmer', type=bool, default=False, 
                        help="Whether to stem the tokens before one-hot-encoding.")
    args = parser.parse_args()

    data = load_data("project/data/Sarcasm_Headlines_Dataset_v2.json")
    data_samples_and_stats(data)
    X_train, X_test, y_train, y_test = split_data(data)

    for n in args.top_tokens:
        # One-hot encoding for the given number of top tokens
        X_train_one_hot, X_test_one_hot, feature_names = one_hot_encoding(n, X_train, X_test, args.stemmer)
        
        # Print some samples of the one hot encoded data
        X_train_encoded_df = pd.DataFrame(X_train_one_hot.toarray(), columns=feature_names, index=X_train)
        print(X_train_encoded_df.sample(n=5))

        # Save the data with one-hot encoding for this specific n top tokens
        save_data_with_one_hot(
            X_train, X_test, y_train, y_test, 
            X_train_one_hot, X_test_one_hot,
            feature_names, data.loc[X_train.index], data.loc[X_test.index],
            "project/data/splits", args.stemmer
        )


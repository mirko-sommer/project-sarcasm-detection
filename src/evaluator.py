from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tabulate import tabulate

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def specificity_score(y_true, y_pred):
    """
    Calculate the specificity score (True Negative Rate) for binary classification.

    Args:
        y_true (array-like): True class labels for the samples.
        y_pred (array-like): Predicted class labels for the samples.

    Returns:
        float: The specificity score.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)


class Evaluator:
    def __init__(self, model, model_name, metrics=None):
        """
        Initialize the evaluator with a model and evaluation metrics.

        Args:
            model (object): The machine learning model to be evaluated.
            metrics (dict, optional): A dictionary of evaluation metrics with their corresponding functions.
                                      Default is accuracy, F1-score, and specificity.
        """
        self.model = model
        self.model_name = model_name
        self.metrics = metrics or {
            "accuracy": accuracy_score,
            "f1": lambda y_true, y_pred: f1_score(y_true, y_pred, zero_division=1),
            "specificity": specificity_score,
            "sensitivity": lambda y_true, y_pred: recall_score(y_true, y_pred, zero_division=1),  
        }
        self.results = {}
        self.summarized_results = None
        self.results_to_summarize = ["train", "dev"]


    def evaluate(self, X, y, set_name, cv_iteration=0):
        """
        Evaluate the model on a given set and store the results for each fold.

        Args:
            X (array-like): The feature matrix for the data.
            y (array-like): The true class labels for the data.
            set_name (str): The name of the dataset (e.g., 'train', 'dev', 'test').
            cv_iteration (int): The current iteration of cross-validation.

        Returns:
            dict: A dictionary containing the evaluation results for each metric.
        """
        y_pred = self.model.predict(X)
        
        results = {
            metric: scorer(y, y_pred)
            for metric, scorer in self.metrics.items()
        }
        
        if cv_iteration not in self.results:
            self.results[cv_iteration] = {}

        self.results[cv_iteration][set_name] = results

        return results
    

    def summarize_results(self):
        """
        Calculate the mean and standard deviation for each metric over the cross-validation folds.

        Returns:
            dict: A dictionary containing the summarized results (mean and std) for each set and metric.
        """
        self.summarized_results = {
            set_name: {
                metric: {
                    "mean": np.mean([self.results[fold_idx][set_name][metric] 
                                     for fold_idx in self.results if set_name in self.results[fold_idx]]),
                    "std": np.std([self.results[fold_idx][set_name][metric] 
                                   for fold_idx in self.results if set_name in self.results[fold_idx]])
                }
                for metric in self.metrics
            }
            for set_name in self.results_to_summarize
        }

        return self.summarized_results
    

    def print_cv_results_table(self, only_mean=False):
        """
        Print a table of the cross-validation results, showing metrics for each fold.

        Args:
            only_mean (bool): If True, only the mean ± standard deviation for each set is displayed. 
                              If False, individual fold results are also included.
        """
        headers = ["Iteration", "Set"] + list(self.metrics.keys())

        table_data = []

        if only_mean is False:
            for fold_idx, fold_results in self.results.items():
                for set_name, results in fold_results.items():
                    if set_name != "test":
                        row = [fold_idx, set_name] + [f"{value:.4f}" for value in results.values()]
                        table_data.append(row)
        
        if self.summarized_results is None:
            self.summarize_results()

        for set_name in self.results_to_summarize:
            summarized_row = ["Mean ± Deviation"] + [set_name] + [
                f"{self.summarized_results[set_name][metric]['mean']:.4f}" + " ± " + f"{self.summarized_results[set_name][metric]['std']:.4f}" 
                for metric in self.metrics
            ]
            table_data.append(summarized_row) 
                    
        print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))


    def print_test_results_table(self):
        """
        Print a table of the test set results for each fold.

        Displays the metrics for the test set, after training the model on each fold.

        """     
        headers = ["Set"] + list(self.metrics.keys())
        
        table_data = []
        for fold_results in self.results.values(): 
            if "test" in fold_results:
                test_row = ["test"] + [f"{float(value):.4f}" for value in fold_results["test"].values()]
                table_data.append(test_row)

        print(tabulate(table_data, headers=headers, tablefmt="fancy_grid", disable_numparse=True))
    

    def visualize_with_tsne(self, X, y, plot_path="project/results/plots", pca_components=30, tsne_perplexity=30):
        """
        Create side-by-side plots for the true and predicted values using dimensional reduction techniques.

        Args:
            X (array-like): The feature matrix.
            y (array-like): The target values.
            plot_path (str): The directory where the plot will be saved.
            pca_components (int): The number of components used for the PCA reduction.
            tsne_perplexity (int): The perplexity for the t-SNE reduction.
        """
        y_pred = self.model.predict(X)

        print("Reducing dimensions with PCA for visualization...")
        pca = PCA(n_components=pca_components) 
        X_pca = pca.fit_transform(X)

        print("Reducing dimensions with t-SNE for visualization...")
        tsne = TSNE(n_components=2, perplexity=tsne_perplexity, random_state=42)
        X_embedded = tsne.fit_transform(X_pca)

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        classes = np.unique(y)
        class_names = ['non-sarcastic', 'sarcastic']
        colors = ['green', 'red']  

        # True labels plot
        for cls, color in zip(classes, colors):
            mask = y == cls
            axes[0].scatter(X_embedded[mask, 0], X_embedded[mask, 1], label=f"Class {class_names[cls]}", color=color, alpha=0.7)
        axes[0].set_title(f"True Labels ({self.model_name})")
        axes[0].set_xlabel("t-SNE Dimension 1")
        axes[0].set_ylabel("t-SNE Dimension 2")
        axes[0].legend()

        # Predicted labels plot
        for cls, color in zip(classes, colors):
            mask = y_pred == cls
            axes[1].scatter(X_embedded[mask, 0], X_embedded[mask, 1], label=f"Class {class_names[cls]}", color=color, alpha=0.7)
        axes[1].set_title(f"Predicted Labels ({self.model_name})")
        axes[1].set_xlabel("t-SNE Dimension 1")
        axes[1].set_ylabel("t-SNE Dimension 2")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(f"{plot_path}/feature_visualization_{self.model_name.replace(' ', '_')}.png")
        plt.show()
    

    def create_confusion_matrix(self, X, y, feature_names, plot_path="project/results/plots", tokens_to_plot=15):
        """
        Create a confusion matrix showing the probability of the model predicting a specific class (e.g., "sarcastic") 
        based on the co-occurrence of the top tokens in the dataset.

        Args:
            X (array-like): The feature matrix.
            y (array-like): The target values.
            feature_names (list of str): List of feature names corresponding to the columns in X.
            plot_path (str): Directory where the plot will be saved.
            tokens_to_plot (int): The number of top tokens (based on frequency) to include in the confusion matrix.

        Outputs:
            - A heatmap showing probabilities for all pairs of the top `tokens_to_plot` tokens, with NaN cells masked.
        """
        token_frequencies = np.sum(X, axis=0) 
        top_indices = np.argsort(token_frequencies)[::-1][:tokens_to_plot]
        top_features = [feature_names[i] for i in top_indices]

        probabilities = np.full((tokens_to_plot, tokens_to_plot), np.nan)  # Initialize with NaN
        
        y = self.model.predict(X)
        for i, f1 in enumerate(top_indices):
            for j, f2 in enumerate(top_indices):
                f1_values = X[:, f1]
                f2_values = X[:, f2]
                # Select rows where both features are active
                active_rows = (f1_values == 1) & (f2_values == 1)
                num_active_rows = np.sum(active_rows)
                if num_active_rows > 0:
                    probabilities[i, j] = np.mean(y[active_rows] == 1)
                else:
                    probabilities[i, j] = np.nan

        plt.figure(figsize=(12, 10))
        sns.heatmap(probabilities, annot=True, fmt=".2f", cmap="coolwarm",
                    xticklabels=top_features, yticklabels=top_features,
                    mask=np.isnan(probabilities), cbar_kws={'label': 'Probability of Prediction "sarcastic"'})
        plt.title(f"Confusion Matrix: Probability of Prediction 'sarcastic' ({self.model_name})")
        plt.xlabel("Token 1")
        plt.ylabel("Token 2")

        plt.tight_layout()
        plot_file = f"{plot_path}/confusion_matrix_{self.model_name.replace(' ', '_')}.png"
        plt.savefig(plot_file)
        plt.show()
        print(f"Confusion matrix plot saved to {plot_file}")


def compare_evaluators(*evaluators, plot_path="project/results/plots", sets_to_plot=("train", "dev", "test"), title=None, filename="metrics_comparison"):
    """
    Compare evaluators by plotting their metrics for specified datasets.

    Parameters:
        plot_path (str): The directory where the plot will be saved.
        *evaluators: Instances of evaluator objects.
        sets_to_plot (tuple): Datasets to include in the plot (default: "train", "dev", "test").
        title: Changes the title of the plot.
    """
    summarized_results = {}
    for evaluator in evaluators:
        evaluator.summarize_results()
        summarized_results[evaluator.model_name] = evaluator.summarized_results

    metrics = list(evaluators[0].metrics.keys())
    x = np.arange(len(metrics)) 

    num_evaluators = len(evaluators)
    width = 0.8 / (num_evaluators * (len(sets_to_plot)))  
    offset_step = width * (len(sets_to_plot))  

    fig, ax = plt.subplots(figsize=(12, 6 + 0.5 * num_evaluators)) 

    colors = {"train": "lightblue", "dev": "lightgreen", "test": "lightcoral"}
    hatch_patterns = ['/', '+', 'o', '\\', '|', '-', 'x']

    for evaluator_idx, evaluator in enumerate(evaluators):
        summarized = summarized_results[evaluator.model_name]

        means = {set_name: [] for set_name in sets_to_plot}
        stds = {set_name: [] for set_name in sets_to_plot}

        for set_name in sets_to_plot:
            if set_name in summarized:
                for metric in metrics:
                    means[set_name].append(summarized[set_name][metric]["mean"])
                    stds[set_name].append(summarized[set_name][metric]["std"])
            elif set_name == "test" and "test" not in summarized:
                test_means = {metric: [] for metric in metrics}
                test_stds = {metric: [] for metric in metrics}
                for fold in evaluator.results.values():
                    if "test" in fold:
                        for metric in metrics:
                            test_means[metric].append(fold["test"][metric])
                            test_stds[metric].append(fold["test"][metric])

                means["test"] = [np.mean(test_means[metric]) for metric in metrics]
                stds["test"] = [np.std(test_stds[metric]) for metric in metrics]

        offset = evaluator_idx * offset_step

        for idx, set_name in enumerate(sets_to_plot):
            if set_name in means and means[set_name]:
                ax.bar(
                    x + offset + idx * width,
                    means[set_name],
                    width,
                    yerr=stds[set_name],
                    label=f"{evaluator.model_name} - {set_name.capitalize()}",
                    color=colors.get(set_name, "gray"),
                    hatch=hatch_patterns[evaluator_idx % len(hatch_patterns)],
                    capsize=5
                )

    ax.set_xlabel("Metrics")
    ax.set_ylabel("Scores")

    if title is None:
        ax.set_title("Comparison of Metrics Across Models and Sets")
    else: 
        ax.set_title(title)

    group_center = (num_evaluators * offset_step) / 2 - (width * 0.5)
    ax.set_xticks(x + group_center)
    ax.set_xticklabels(metrics)
    plt.legend(loc='lower left')

    plt.tight_layout()
    plt.savefig(f"{plot_path}/{filename.replace(' ', '_')}.png")
    plt.show()


def plot_feature_importances(model, feature_names, model_name, plot_path="project/results/plots",):
    """
    Plot the feature importances of a trained model.

    Args:
        plot_path (str): The directory where the plot will be saved.
        model: A trained model with a feature_importances_ attribute.
        feature_names (list): List of feature names.
        model_name (str): Name of the model to use in the title.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]  # Sort in descending order

    plt.figure(figsize=(15, 6))
    plt.title(f"Feature Importances - {model_name}")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), np.array(feature_names)[indices], rotation=90)
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(f"{plot_path}/{model_name.replace(' ', '_')}_Feature_Importance.png")
    plt.show()

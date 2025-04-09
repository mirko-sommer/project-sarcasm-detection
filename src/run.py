import argparse

import trainer_tester as trainer_tester
from trainer_tester import train_and_evaluate_model
from evaluator import compare_evaluators
from data import load_preprocessed_data
from baseline import baseline_random, baseline_random_distribution_of_train, baseline_most_frequent

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from itertools import zip_longest

import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sarcasm Headline Classification")
    parser.add_argument('--model', type=str, choices=['DecisionTree', 'RandomForest', 'MLP', 'SVM'], 
                        help="The model to use. Choices are: DecisionTree, RandomForest, MLP, SVM.")
    parser.add_argument('--top_tokens', type=int, default=100, 
                        help="Number of top tokens (features) to use for baseline/fine-tune (Has to be created in advance with data.py).")
    parser.add_argument('--default_params', type=bool, default=False, 
                        help="Whether to fit the model with default parameters")
    parser.add_argument('--fine_tune', type=bool, default=False, 
                        help="Whether to fine-tune the model.")
    parser.add_argument('--stemmer', type=bool, default=False, 
                        help="Whether to use the stemmed dataset / Include stemmed dataset in comparison")
    parser.add_argument('--test', type=bool, default=False, 
                        help="Whether to test the model on the test set. (Only the results on the test set are shown.)")
    
    parser.add_argument('--compare', type=bool, default=False, 
                        help="Whether to create comparison for different top tokens/stemming.")
    parser.add_argument('--top_tokens_list', type=int, nargs='+', default=[100, 200, 300, 400, 500, 1000, 1500], 
                        help="List of the number of top tokens (features) to use for comparison (Dataset have to be created in advance).")
    
    parser.add_argument('--models_to_compare', type=str, nargs='+', choices=['DummyRandom', 'DummyRandomDistributionTrain', 'DummyMajority', 'DecisionTree', 'RandomForest', 'MLP', 'SVM'], default=None,  
                        help="List models to compare.")
    parser.add_argument('--models_top_token_list', type=int, nargs='+', 
                        help="List of the number of top tokens to use for each model.")
    parser.add_argument('--models_n_splits_list', type=int, nargs='+', 
                        help="List of the number of splits to use for each model. Default 4 splits for each model.")
    
    parser.add_argument('--feature_importance', type=bool, default=False, 
                        help="Whether to plot feature importance for the models.")
    
    parser.add_argument('--visualize_features', type=bool, default=False, 
                        help="Whether to create a plot of the features using dimensional reduction techniques.")
    parser.add_argument('--confusion_matrix', type=bool, default=False, 
                        help="Whether to create a confusion matrix showing the probability of the model predicting a specific class based on the co-occurrence of the top tokens in the dataset.")
    
    args = parser.parse_args()

    # Enforce at least one of --model or --models_to_compare is provided
    if not args.model and args.models_to_compare is None:
        raise ValueError("You must specify either --model or --models_to_compare.")
    if args.model and args.models_to_compare:
        raise ValueError("You cannot specify both --model and --models_to_compare.")

    model_map = {
    "DecisionTree": DecisionTreeClassifier,
    "RandomForest": RandomForestClassifier,
    "MLP": MLPClassifier,
    "SVM": SVC
    }

    # Define hyperparameter grids
    param_map = {
    "DecisionTree": {
                    'max_depth': [50, 100, 200, None],
                    'min_samples_split': [3, 5, 10],
                    'min_samples_leaf': [5, 10, 20, 50],
                    'criterion': ['gini', 'entropy']
                    },
    "RandomForest": {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 50, 100, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'criterion': ['gini', 'entropy']
                    },
    "MLP": {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)],
            'activation': ['identity', 'logistic', 'tanh', 'relu'],
            'solver': ['lbfgs', 'sgd', 'adam'],
            'alpha': [0.0001, 0.001, 0.01, 0.1],
            'learning_rate': ['constant', 'invscaling', 'adaptive'],
            'max_iter': [200, 500, 1000]
            },
    "SVM": {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'poly', 'rbf'],
            'gamma': ['scale', 'auto']
            }
    }

    vectorizer = DictVectorizer(sparse=False)

    if not args.compare and args.models_to_compare is None:
        model_cls = model_map[args.model]

        X_train, y_train, X_test, y_test, feature_names = load_preprocessed_data(
            top_tokens=args.top_tokens,
            stemmed=args.stemmer,
            vectorizer=vectorizer
        )
    
    
        if args.default_params:
            # Model with default parameters
            tt_base = train_and_evaluate_model(model_cls, X_train, y_train, X_test, y_test, 
                                               use_test=args.test, load_parameters=False, 
                                               model_name=f"{model_cls.__name__} Base{' stemmed' if args.stemmer else ''}", 
                                               feature_names=feature_names if args.feature_importance else None)
            
            if args.visualize_features: 
                print("Visualizing with t-SNE...")
                tt_base.eval.visualize_with_tsne(X_test, y_test)

            if args.confusion_matrix:
                tt_base.eval.create_confusion_matrix(X_test, y_test, feature_names)


        if args.fine_tune:
            # Fine-tune model
            tt_fine_tuned = train_and_evaluate_model(model_cls, X_train, y_train, X_test, y_test, 
                                                     use_test=args.test, param_grid=param_map[args.model], 
                                                     model_name=f"{model_cls.__name__} fine-tuned{' stemmed' if args.stemmer else ''}", 
                                                     feature_names=feature_names if args.feature_importance else None)

            if args.visualize_features: 
                print("Visualizing with t-SNE...")
                tt_fine_tuned.eval.visualize_with_tsne(X_test, y_test)

            if args.confusion_matrix:
                tt_fine_tuned.eval.create_confusion_matrix(X_test, y_test, feature_names)

        if args.default_params and args.fine_tune:
            # Compare Default model with fine-tuned model
            compare_evaluators(tt_base.eval, tt_fine_tuned.eval, sets_to_plot=("test", ) if args.test else ("train", "dev"), 
                               title=f"Comparison {model_cls.__name__} base -> fine-tuned{' stemmed' if args.stemmer else ''}{' on test' if args.test else ''}",
                               filename=f"Comparison {model_cls.__name__} base-fine-tuned{'_stemmed' if args.stemmer else ''}{'_test' if args.test else ''}")
        
    if args.compare:
        # Compare models for different max token counts
        model_cls = model_map[args.model]

        models = []
        for max_token in args.top_tokens_list:
            for s in [False, True] if args.stemmer else [False]:
                X_train, y_train, X_test, y_test, feature_names = load_preprocessed_data(
                    top_tokens=max_token,
                    stemmed=s,
                    vectorizer=vectorizer
                )
                model = train_and_evaluate_model(model_cls, X_train, y_train, X_test, y_test, 
                                                 show_plot=False, 
                                                 model_name=f"{model_cls.__name__} top {max_token} words {'stemmed' if s else ''}")
                
                models.append(model)

        compare_evaluators(
            *[m.eval for m in models],
            sets_to_plot=("dev", ),
            title=f"Comparison of {model_cls.__name__} for different top n tokens",
            filename=f"Comparison {model_cls.__name__} for different top n tokens{' stemmed' if args.stemmer else ''}"
        )
    

if args.models_to_compare is not None:
    # Ensure models and top token lists are of the same length
    if len(args.models_to_compare) != len(args.models_top_token_list):
        raise ValueError("Length of --models_to_compare must match --models_top_token_list.")
    
    results = []

    models_n_splits_list = args.models_n_splits_list if args.models_n_splits_list is not None else []

    for model, top_token, n_split in zip_longest(args.models_to_compare, args.models_top_token_list, models_n_splits_list, fillvalue=None):
        # Load and preprocess data
        X_train, y_train, X_test, y_test, feature_names = load_preprocessed_data(
            top_tokens=top_token,
            stemmed=args.stemmer,
            vectorizer=vectorizer
        )

        if model == "DummyRandom":
            tt_baseline_random = baseline_random(X_train, y_train, X_test, y_test, show_plot=False, model_name=f"Dummy Random (top {top_token} tokens){' stemmed' if args.stemmer else ''}")
            results.append(tt_baseline_random)
        elif model == "DummyRandomDistributionTrain":
            tt_baseline_random_distribution_of_train = baseline_random_distribution_of_train(X_train, y_train, X_test, y_test, show_plot=False, model_name=f"Dummy Random with Distribution of Train (top {top_token} tokens){' stemmed' if args.stemmer else ''}")
            results.append(tt_baseline_random_distribution_of_train)
        elif model == "DummyMajority":
            tt_baseline_majority = baseline_most_frequent(X_train, y_train, X_test, y_test, show_plot=False, model_name=f"Dummy Majority (top {top_token} tokens){' stemmed' if args.stemmer else ''}")
            results.append(tt_baseline_majority)
        else:
            model_cls = model_map[model]
            # Train and evaluate model
            tt = train_and_evaluate_model(
                model_cls,
                X_train,
                y_train,
                X_test,
                y_test,
                n_splits=n_split,
                use_test=args.test,
                load_parameters=(False if args.default_params else True),
                show_plot=False,
                model_name=f"{model_cls.__name__} (top {top_token} tokens){' stemmed' if args.stemmer else ''}{f' {n_split}' if n_split is not None else ''}{' splits' if n_split is not None else ''}"
            )

            results.append(tt)

    # Compare results across models
    compare_evaluators(
        *[m.eval for m in results],
        sets_to_plot=("test", ) if args.test else ("dev", ),
        title=f"Comparison of Models: {', '.join(args.models_to_compare)}",
        filename=f"Comparison_{'_'.join(args.models_to_compare)}{'_stemmed' if args.stemmer else ''}{'_test' if args.test else ''}"
    )
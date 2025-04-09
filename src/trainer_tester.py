import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from evaluator import Evaluator, compare_evaluators, plot_feature_importances
from save_load import load_params, save_params

class TrainerTester:
    def __init__(self, model, cv=4, model_name=None):
        """
        Initialize the TrainerTester with a model, cross-validation settings, and an evaluator.

        Args:
            model (object): The machine learning model to be trained and tested.
            cv (int, optional): The number of cross-validation folds. Default is 4.
            model_name (str, optional): The name of the model. If not provided, it defaults to the model's class name.
        """
        self.model = model
        self.cv = cv
        self.model_name = model_name or self.model.__class__.__name__    
        self.eval = Evaluator(model=self.model, model_name=self.model_name)

    def cv_train(self, X, y, n_splits=4):
        """
        Perform cross-validation training on the model, evaluating on each fold.

        Args:
            X (array-like): The feature matrix for the dataset.
            y (array-like): The true class labels for the dataset.
            n_splits (int): The number of folds. (default 4)

        This method splits the data into `cv` (default 4) folds, trains the model on each fold's training data, 
        and evaluates the model on both training and development sets for each fold.
        """
        X, y = np.array(X), np.array(y)

        kf = StratifiedKFold(n_splits=n_splits)

        for iteration, (train_fold, dev_fold) in enumerate(kf.split(X, y)):
            X_train, y_train = X[train_fold], y[train_fold]
            X_dev, y_dev = X[dev_fold], y[dev_fold]

            self.model.fit(X_train, y_train)

            self.eval.evaluate(X_train, y_train, set_name="train", cv_iteration=iteration)
            self.eval.evaluate(X_dev, y_dev, set_name="dev", cv_iteration=iteration)
    
    def fine_tune_model(self, X_train, y_train, param_grid, scoring="f1"):
        """
        Fine-tune the model's hyperparameters using GridSearchCV.

        Args:
            X_train (array-like): The training feature matrix.
            y_train (array-like): The training target values.
            param_grid (dict): A dictionary with hyperparameters to search over.
            scoring (str, optional): The metric to use for scoring during the grid search. Default is 'accuracy'.

        Returns:
            Best model (object): The best model after the grid search.
        """
        grid_search = GridSearchCV(estimator=self.model, param_grid=param_grid, 
                                   scoring=scoring, cv=self.cv, verbose=1)
        grid_search.fit(X_train, y_train)

        print(f"Best parameters found by GridSearchCV: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_}")

        # Set the model to the best estimator found by GridSearchCV
        self.model = grid_search.best_estimator_

        return self.model
    

    def train(self, X, y):
        """
        Train the model on the entire dataset.

        Args:
            X (array-like): The feature matrix for the dataset.
            y (array-like): The true class labels for the dataset.

        This method fits the model on the full dataset after cross-validation.
        """
        self.model.fit(X, y)


    def test(self, X, y):
        """
        Test the model on a dataset and evaluate the results.

        Args:
            X (array-like): The feature matrix for the test set.
            y (array-like): The true class labels for the test set.

        This method evaluates the model on the test set after training.
        """
        self.eval.evaluate(X, y, set_name="test")


    def print_result_tables(self, test=True, only_mean=False):
        """
        Print the evaluation result tables for the cross-validation and test sets.

        Args:
            test (bool, optional): If True, prints the test results. Default is True.
            only_mean (bool, optional): If True, only prints the mean values for each metric. Default is False.

        This method prints tables for cross-validation results, and optionally, test set results.
        """
        print(f"Result tables for {self.model_name}: ")
        self.eval.print_cv_results_table(only_mean=only_mean)
        if test:
            self.eval.print_test_results_table()


def train_and_evaluate_model(model_class,
                             X_train, y_train, X_test, y_test, n_splits=4,
                             param_grid=None, 
                             use_test=False, show_plot=True, model_name=None, 
                             load_parameters=True, 
                             feature_names=None):
    """
    Trains and evaluates a model.

    Args:
        model_class(sklearn model): Which model class to use. E.g. DecisionTreeClassifier
        X_train (array-like): Training feature set.
        y_train (array-like): Training target labels.
        X_test (array-like): Test feature set.
        y_test (array-like): Test target labels.
        n_splits (int): The number of folds to split the train data.
        param_grid (dict, optional): Hyperparameter grid for fine-tuning. Defaults to None.
        use_test (bool, optional): Whether to evaluate on the test set. Defaults to False.
        show_plot (bool, optional): Whether to generate evaluation plots. Defaults to True.
        model_name (str, optional): Name for the model. Defaults to model class.
        load_parameters (bool, optional): Whether to load pre-saved parameters. Defaults to True.
        feature_names(list): List of all feature names. If feature names are given a feature importance plot is generated.

    Returns:
        TrainerTester: The trained and evaluated model wrapper.
    """
    n_splits = (4 if n_splits is None else n_splits)
    
    model = model_class()
    trainer = TrainerTester(model=model, model_name=model_name)
    if model_name is None:
        model_name = f"{model.__class__.__name__}"
    
    if load_parameters:
        # Try loading saved parameters
        saved_params = load_params(f"{model.__class__.__name__}")
    else:
        saved_params = None

    if saved_params:
        model.set_params(**saved_params)
        trainer = TrainerTester(model=model, model_name=f"{model_name} (Loaded Params)")
    elif param_grid:
        model = trainer.fine_tune_model(X_train, y_train, param_grid)
        trainer = TrainerTester(model=model, model_name=f"{model_name} Fine-tuned")
        save_params(model.get_params(), f"{model.__class__.__name__}")

    trainer.cv_train(X_train, y_train, n_splits=n_splits)

    if use_test:
        trainer.train(X_train, y_train)
        trainer.test(X_test, y_test)

    trainer.print_result_tables(test=use_test, only_mean=True)
    if show_plot:
        if use_test:
            compare_evaluators(trainer.eval, sets_to_plot=("test", ), title=f"{model_name}", filename=f"{model_name}_test".replace(" ", "_"))
        else:
            compare_evaluators(trainer.eval, sets_to_plot=("train", "dev"), title=f"{model_name}", filename=f"{model_name}".replace(" ", "_"))
    
    if feature_names is not None:
        plot_feature_importances(model, feature_names, model_name)

    return trainer
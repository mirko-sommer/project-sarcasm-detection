from data import load_data
from sklearn.dummy import DummyClassifier
import trainer_tester as trainer_tester
from evaluator import compare_evaluators


def baseline_random(X_train, y_train, X_test, y_test, show_plot=True, model_name="DummyClassifier - Random"):
    """
    Train and evaluate a baseline model using a DummyClassifier with random predictions.

    Args:
        X_train (array-like): The feature matrix for the training data.
        y_train (array-like): The target values for the training data.
        X_test (array-like): The feature matrix for the test data.
        y_test (array-like): The target values for the test data.
        show_plot (bool): Defines if the plot is shown.
        model_name (str): Name of the model used for the plot and results.
    """
    dummy_clf = DummyClassifier(strategy="uniform", random_state=42)
    trainer_tester_dummy = trainer_tester.TrainerTester(model=dummy_clf, model_name=model_name)
    
    trainer_tester_dummy.cv_train(X_train, y_train)

    trainer_tester_dummy.train(X_train, y_train) # Retrain the model on the full dataset after cross validation
    trainer_tester_dummy.test(X_test, y_test)
    
    trainer_tester_dummy.print_result_tables()
    if show_plot:
        compare_evaluators(trainer_tester_dummy.eval, sets_to_plot=("train", "dev", "test"), title=f"Comparison of {model_name}", filename=f"Comparison {model_name}")

    return trainer_tester_dummy


def baseline_random_distribution_of_train(X_train, y_train, X_test, y_test, show_plot=True, model_name="DummyClassifier - Random Distribution of Train"):
    """
    Train and evaluate a baseline model using a DummyClassifier with stratified random predictions.

    Args:
        X_train (array-like): The feature matrix for the training data.
        y_train (array-like): The target values for the training data.
        X_test (array-like): The feature matrix for the test data.
        y_test (array-like): The target values for the test data.
        show_plot (bool): Defines if the plot is shown.
        model_name (str): Name of the model used for the plot and results.
    """

    dummy_clf = DummyClassifier(strategy="stratified", random_state=42)
    trainer_tester_dummy = trainer_tester.TrainerTester(model=dummy_clf, model_name=model_name)
    
    trainer_tester_dummy.cv_train(X_train, y_train)

    trainer_tester_dummy.train(X_train, y_train) # Retrain the model on the full dataset after cross validation
    trainer_tester_dummy.test(X_test, y_test)

    trainer_tester_dummy.print_result_tables()
    if show_plot:
        compare_evaluators(trainer_tester_dummy.eval, sets_to_plot=("train", "dev", "test"), title=f"Comparison of {model_name}", filename=f"Comparison {model_name}")
    
    return trainer_tester_dummy


def baseline_most_frequent(X_train, y_train, X_test, y_test, show_plot=True, model_name="DummyClassifier - Most Frequent"):
    """
    Train and evaluate a baseline model using a DummyClassifier with most frequent class prediction.

    Args:
        X_train (array-like): The feature matrix for the training data.
        y_train (array-like): The target values for the training data.
        X_test (array-like): The feature matrix for the test data.
        y_test (array-like): The target values for the test data.
        show_plot (bool): Defines if the plot is shown.
        model_name (str): Name of the model used for the plot and results.
    """
    dummy_clf = DummyClassifier(strategy="most_frequent")
    trainer_tester_dummy = trainer_tester.TrainerTester(model=dummy_clf, model_name=model_name)
    
    trainer_tester_dummy.cv_train(X_train, y_train)

    trainer_tester_dummy.train(X_train, y_train) # Retrain the model on the full dataset after cross validation
    trainer_tester_dummy.test(X_test, y_test)

    trainer_tester_dummy.print_result_tables()
    if show_plot:
        compare_evaluators(trainer_tester_dummy.eval, sets_to_plot=("train", "dev", "test"), title=f"Comparison of {model_name}", filename=f"Comparison {model_name}")

    return trainer_tester_dummy
    

if __name__ == "__main__":
    test_data = load_data("project/data/splits/not_stemmed/100_test_data.json")
    train_data = load_data("project/data/splits/not_stemmed/100_train_data.json")

    X_test = test_data["one_hot_encoding"]
    y_test = test_data["is_sarcastic"]

    X_train = train_data["one_hot_encoding"]
    y_train = train_data["is_sarcastic"]

    tt_baseline_random = baseline_random(X_train, y_train, X_test, y_test)
    tt_baseline_random_distribution_of_train = baseline_random_distribution_of_train(X_train, y_train, X_test, y_test)
    tt_baseline_most_frequent = baseline_most_frequent(X_train, y_train, X_test, y_test)

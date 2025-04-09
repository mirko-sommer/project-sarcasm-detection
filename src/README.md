# Code for Project

## 🚀 Description

This directory contains all the code that is needed for this project. The README contains all the information about how to run and use the code.

## 📑 Data splits, feature extraction and dataset statistics

The dataset has two distinct version from which we will use the second one. The dataset has to be downloaded manually (as it is to big) and be placed into "./data/Sarcasm_Headlines_Dataset_v2.json".

| Argument | Description | Choices | Info | 
| --- | --- | --- | --- |
| `--top_tokens` | Number of top tokens (features) to include for one-hot encoding. | int, list | Default: 100 |
| `--stemmer` | Whether to stem the tokens before one-hot-encoding. | bool | Default: False |

Example call to create test/train splits, one-hot-encoding and show some dataset samples and statistics:
```python
python data.py --top_tokens 100 200 300 400 500 1000 1500 2000 --stemmer False
```

### Output
The test and train splits, with corresponding one-hot-encodings are saved under [../data/splits/](../data/splits/).

## 📊 Class: Evaluator
The class Evaluator in the script [evaluator.py](./evaluator.py) provides a comprehensive evaluation framework to assess model performance using multiple metrics. It supports cross-validation, result summarization and visual representation of metrics.

For further details of how to use the class refer to the respective docstrings.

## 🤖 Class: TrainerTester
The class TrainerTester in the script [trainer_tester.py](./trainer_tester.py) simplifies the process of training machine learning models for our task by performing cross-validation and evaluating results on train, dev and test sets using the Evaluator class to calculate metrics, print results and generate results.

For further details of how to use the class refer to the respective docstrings.

## 🏁 Baseline generation
The script [baseline.py](./baseline.py) provides a baseline performance benchmark for our sarcasm detection model using 'DummyClassifier' from 'sklearn'. 

Example call to create baseline results:
```python
python baseline.py
```

The resulting plots are saved under [../results/plots](../results/plots/).

## 🌳 Code to create results

The script [run.py](./run.py) provides the possibility to train, evaluate, hyperparameter-tune, analyze and compare our different models on the dataset.

| Argument | Description | Choices | Info | 
| --- | --- | --- | --- |
| `--model` | The model to use. Choices are: DecisionTree, RandomForest, MLP, SVM. | str |  |
| `--top_tokens` | Number of top tokens (features) to use for baseline/hyperparameter-tune (Has to be created in advance with [data.py](./data.py)). | int | Default: 100 |
| `--default_params` | Whether to fit the model with default parameters | bool | Default: False |
| `--fine_tune` | Whether to hyperparameter-tune the model. (or load the hyperparameter-tuned model) | bool | Default: False |
| `--stemmer` | Whether to use the stemmed dataset / Include stemmed dataset in comparison | bool | Default: False |
| `--test` | Whether to test the model on the test set. (Only the results on the test set are shown.) | bool | Default: False |
| | | | |
| `--compare` | Whether to create comparison for different top tokens/stemming. | bool | Default: False |
| `--top_tokens_list` | List of the number of top tokens (features) to use for comparison (Dataset have to be created in advance with [data.py](./data.py)). | list of ints | Default: [100, 200, 300, 400, 500, 1000, 1500] |
| | | | |
| `--models_to_compare` | List models to compare. Models are compared if parameter is given. | list of strings | Choices are: 'DummyRandom', 'DummyRandomDistributionTrain', 'DummyMajority', 'DecisionTree', 'RandomForest', 'MLP', 'SVM' |
| `--models_top_token_list` | List of the number of top tokens to use for each model. | list of ints | |
| `--models_n_splits_list` | List of the number of splits to use for each model. Default 4 splits for each model. | list of ints | |
| | | | |
| `--feature_importance` | Whether to plot feature importance for the models. | bool | Default: False |
| | | | |
| `--visualize_features` | Whether to create a plot of the features using dimensional reduction techniques. | bool | Default: False |
| `--confusion_matrix` | Whether to create a confusion matrix showing the probability of the model predicting a specific class based on the co-occurrence of the top tokens in the dataset. | bool | Default: False|

Example call to train, evaluate and analyze decision tree with default parameters on 200 top token dataset:
```python
python run.py --model DecisionTree --top_tokens 200 --default_params True
```

Example call to train, hyperparameter-tune, evaluate and analyze random forest with default parameters on 200 top token stemmed dataset:
```python
python run.py --model RandomForest --top_tokens 200 --fine_tune True --stemmer True
```

Example call to create comparison plot for different top tokens (100, 200, 300 top tokens) with stemming:
```python
python run.py --model DecisionTree --compare True --top_tokens_list 100 200 300 --stemmer True
```

Example call to create feature importance plot on hyperparameter-tuned decision tree on 100 top token dataset:
```python
python run.py --model DecisionTree --feature_importance True --fine_tune True 
```

Example call to create comparison plot of hyperparameter-tuned decision tree on 300 token dataset with hyperparameter-tuned random forest on 1000 token dataset. Both datasets are stemmed.
```python
python run.py --models_to_compare DecisionTree RandomForest --models_top_token_list 300 1000 --stemmer True
```

Example call to create a visualization of the features using a confusion matrix and dimensional reduction techniques.
```python
python run.py --model MLP --top_tokens 2000 --fine_tune True --stemmer True --confusion_matrix True --visualize_features True
```

Example call to create comparison plot of hyperparameter-tuned decision tree on 300 token dataset with hyperparameter-tuned random forest on 1000 token dataset. Both datasets are stemmed. The results and plots are created on the test dataset.
```python
python run.py --models_to_compare DecisionTree RandomForest --models_top_token_list 300 1000 --stemmer True
```

## 🤖 Significance test

The script [significance_test.py](./significance_test.py) provides the possibility to calculate the significance values for the comparison of Baseline vs. Decision Tree vs. Random Forest vs. MLP.



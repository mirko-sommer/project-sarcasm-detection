# Project Sarcasm Detection

## Abstract

This project uses different classifiers (Decision Tree, Random Forest, Multi-Layer Perceptron, Support Vector Machine) for sarcasm detection using sklearn. Training and testing are performed on a dataset for sarcasm detection. The hope is to help people who have difficulty differentiating sarcastic from non-sarcastic sentences, like autistic and neurodivergent people.

**For a detailed report of my methods and results, with tables and plots, please refer to [./report.md](./report.md).**

## 📚 How to run the code
### Requirements

- Python 3.11 / 3.10

To run the scripts in this project, ensure you have the required packages and modules installed. You can find them listed in the [requirements.txt](requirements.txt) file. Create a virtual environment and install the requirements using the following command:

```
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 🚧 TODO
* [x] Evaluation - Train/Dev/Test
    * [x] Migration from ipynb to py
* [x] Baselines
    * [x] Migrate to new plotting functionality?
* [x] Decision Tree, Random Forest
    * [x] Better naming of raw plots
    * [x] Add tree.py to src/README.md
    * [x] Better modularization for future models
    * [x] Add all results tables to appendix (for comparison)
* [x] MLP
    * [x] Calculate significance
* [x] Project-Specifics
    * [x] Create visualization for comparison of different features (using PCA, ...?)
    * [x] Try other classifiers? + Test (SVM?)
* [x] Etc.
    * [x] Move tables to separate folder
    * [x] Add appendix with less important results
    * [x] Add missing tables
* [x] Test models
* [x] Presentation of results
* [x] Finish writing and update documentation/README
    * [x] Write more about methods used
    * [x] Add all TODOs (analysis etc.)
    * [x] Check all docstrings

## ⏳ Future Work

If we had unlimited resources for our project, we would: 
* [ ] Expand the models with other datasets. Possibly even create my own dataset that is better suited for sarcasm.
* [ ] Additionally, we would use better models that are tailored to language, such as LSTM, RNN, LLMs, etc
* [ ] Create more coherent and beautiful tables

## 💁‍♂️ Author

* Mirko Sommer 

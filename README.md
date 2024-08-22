# Comments Sentiment Analysis


## Introduction
Comments Sentiment Analysis is a project focused on analyzing the sentiment of user comments. It utilizes natural language processing (NLP) techniques to classify comments as positive, negative, or neutral. This project aims to provide insights into user opinions and feedback by automatically categorizing the sentiment of their comments.

## Table of Contents


- [Directory Structure](#directory-structure)
- [Files and Functions](#files-and-functions)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [Installation Guide](#installation-guide)
- [Acknowledgments](#acknowledgments)
- [Further Improvements](#further-improvements)
- [License](#license)


## Directory Structure
```
├── src
│ ├── utils.py
│ ├── model_training.py
│ ├── model_evaluation.py
│ ├── data_preprocessing.py
│ └── data_exploration.py
├── notebooks
│ ├── data_exploration.ipynb
│ ├── data_preprocessing.ipynb
│ ├── model_training.ipynb
│ └── model_evaluation.ipynb
├── environment.yml
└── README.md
```


## Files and Functions

- `utils.py` : Utility functions for various tasks.
- `model_training.py` : Functions for training the model.
- `model_evaluation.py` : Functions for evaluating the model.
- `data_preprocessing.py` : Functions for data preprocessing.
- `data_exploration.py` : Functions for data exploration.
- `data_exploration.ipynb`: Notebook for data exploration.
- `data_preprocessing.ipynb`: Notebook for data preprocessing.
- `model_training.ipynb`: Notebook for model training.
- `model_evaluation.ipynb`: Notebook for model evaluation.


## Dataset

The dataset used is the imdb comment Dataset. get the dataset using the fallowing link https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

## Model Performance

### Train
- loss: 0.3185
- accuracy: 0.9040
  
### Validataion
- loss: 0.4368
- accuracy: 0.8587
  
### Test

```
              precision    recall  f1-score   support

           0       0.87      0.85      0.86       376
           1       0.85      0.88      0.86       374

    accuracy                           0.86       750
   macro avg       0.86      0.86      0.86       750
weighted avg       0.86      0.86      0.86       750
```
-  test_loss: 0.44
- accuracy: 0.86
- precision: 0.86
- recall: 0.86
- f1: 0.86
  
## Installation Guide

To set up the project environment, use the `environment.yml` file to create a conda environment.

1. **Clone the repository:**

    ```bash
    git clone https://github.com/sadegh15khedry/Comments-Sentiment-Analysis.git
    cd Comments-Sentiment-Analysis
    ```

2. **Create the conda environment:**

    ```bash
    conda env create -f environment.yml
    ```

3. **Activate the conda environment:**

    ```bash
    conda activate comments
    ```

4. **Verify the installation:**

    ```bash
    python --version
    ```


## Acknowledgments
- Special thanks to the developers and contributors the libraries used in this project, including NumPy, pandas, scikit-learn, Seaborn, and Matplotlib.
- Huge thaks to contributors of the IMDB Dataset.

## Further Improvements

- more hyperparameter tuning to optimize the model parameters.


  
## License
This project is licensed under the MIT License. See the LICENSE file for details.



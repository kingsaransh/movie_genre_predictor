# Movie Genre Prediction Model

This project implements a machine learning model that predicts movie genres based on plot summaries using various classification techniques.

## Features

- Text preprocessing using NLTK
- TF-IDF feature extraction
- Multiple classifier implementations:
  - Naive Bayes
  - Logistic Regression
  - Support Vector Machine (SVM)
- Multi-label classification support
- Model evaluation and comparison

## Requirements

- Python 3.8 or higher
- Required packages are listed in `requirements.txt`

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Dataset

The model expects a dataset in CSV format with at least two columns:
- `plot`: Movie plot summary
- `genres`: Movie genres (pipe-separated string or list)

Place your dataset in a zip file named `dataset.zip` in the project root directory.

## Usage

1. Place your dataset zip file in the project root directory
2. Run the script:
```bash
python movie_genre_predictor.py
```

The script will:
1. Extract and preprocess the dataset
2. Train multiple models
3. Evaluate and compare their performance
4. Display detailed results for each model

## Output

The script provides:
- Overall accuracy for each model
- Detailed classification report
- Comparison of model performances

## Customization

You can modify the following parameters in the code:
- `max_features` in TfidfVectorizer
- Test/train split ratio
- Model hyperparameters
- Text preprocessing steps 
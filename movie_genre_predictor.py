import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import zipfile
import os

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def preprocess_text(text):
    """Preprocess the text data"""
    if not isinstance(text, str):
        return ''
    
    try:
        # Convert to lowercase
        text = text.lower()
        
        # Remove movie titles and years (they're in brackets)
        text = re.sub(r'\[.*?\]', '', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-zA-Z\s.,!?]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Get stopwords
        stop_words = set(stopwords.words('english'))
        
        # Split into words and remove stopwords
        words = text.split()
        words = [word for word in words if word not in stop_words and len(word) > 1]
        
        # If all words were removed, keep the original text
        if not words:
            return text
            
        return ' '.join(words)
    except Exception as e:
        print(f"Error preprocessing text: {str(e)}")
        return text  # Return original text if there's an error

def load_and_preprocess_data(zip_path):
    """Load and preprocess the dataset from zip file"""
    try:
        # Check if zip file exists
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"Zip file not found at: {zip_path}")
            
        print(f"Found zip file at: {zip_path}")
        
        # Extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            print("\nContents of zip file:")
            for file_info in zip_ref.infolist():
                print(f"- {file_info.filename}")
            if not os.path.exists('dataset'):
                os.makedirs('dataset')
            zip_ref.extractall('dataset')
        
        # Path to the data files
        data_dir = os.path.join('dataset', 'Genre Classification Dataset')
        train_file = os.path.join(data_dir, 'train_data.txt')
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"Training data file not found at: {train_file}")
        print(f"\nReading training data from: {train_file}")
        with open(train_file, 'r', encoding='utf-8') as f:
            train_data = f.readlines()
        print("\nFirst 10 lines of train_data.txt:")
        for line in train_data[:10]:
            print(line.strip())
        plots = []
        genres = []
        for line in train_data:
            parts = line.strip().split(' ::: ')
            if len(parts) >= 4:
                plot = parts[3]
                genre = parts[2]
                plots.append(plot)
                genres.append([genre])
        df = pd.DataFrame({
            'plot': plots,
            'genres': genres
        })
        print(f"\nLoaded {len(df)} training examples")
        print("\nSample data:")
        print(df.head())
        df['processed_plot'] = df['plot'].apply(preprocess_text)
        return df
    except Exception as e:
        print(f"\nError loading data: {str(e)}")
        raise

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train and evaluate multiple models"""
    models = {
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'SVM': LinearSVC(max_iter=1000)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate accuracy for each genre
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        
        print(f"\n{name} Results:")
        print(f"Overall Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
    
    return results

def main():
    # Path to your dataset zip file
    zip_path = r"C:/Users/Saransh saini/Downloads/archive.zip"
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data(zip_path)
    
    # Convert genres to single-label format
    y = df['genres'].apply(lambda x: x[0] if isinstance(x, list) else x)
    
    # Create TF-IDF features
    print("Creating TF-IDF features...")
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df['processed_plot'])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train and evaluate models
    print("\nTraining and evaluating models...")
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Print final comparison
    print("\nModel Comparison:")
    for model_name, accuracy in results.items():
        print(f"{model_name}: {accuracy:.4f}")

if __name__ == "__main__":
    main() 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sb

# ==============================
# 1. Load and Preprocess Dataset
# ==============================
def load_and_preprocess_data(file_path):
    """
    Loads the dataset and preprocesses it by handling missing values 
    and converting rainfall into a binary classification.
    """
    data = pd.read_csv(file_path)
    rainfall_threshold = 0.5
    data['Rainfall'] = data['Rainfall'].apply(lambda x: 1 if x > rainfall_threshold else 0)

    # Handling missing values
    for col in data.columns:
        if data[col].isnull().sum() > 0:
            data[col] = data[col].fillna(data[col].mean())

    return data

# ==============================
# 2. Data Visualization
# ==============================
def visualize_data(data):
    """Generates pie chart, histograms, and box plots for data visualization."""
    # Pie chart
    plt.pie(data['Rainfall'].value_counts().values, labels=data['Rainfall'].value_counts().index, autopct='%1.1f%%')
    plt.title("Rainfall Distribution")
    plt.show()

    features = data.select_dtypes(include=np.number).columns
    
    # Histogram plot
    plt.figure(figsize=(15, 8))
    for i, col in enumerate(features[:min(16, len(features))]):
        plt.subplot(4, 4, i + 1)
        sb.histplot(data[col])
    plt.tight_layout()
    plt.show()

    # Box plot
    plt.figure(figsize=(15, 8))
    for i, col in enumerate(features[:min(16, len(features))]):
        plt.subplot(4, 4, i + 1)
        sb.boxplot(y=data[col])
    plt.tight_layout()
    plt.show()

# ==============================
# 3. Model Training and Evaluation
# ==============================
def train_and_evaluate_model(data):
    """
    Trains a Decision Tree Classifier using selected features and evaluates its performance.
    """
    features = ['Temperature', 'Humidity', 'WindSpeed']
    target = 'Rainfall'

    X = data[features]
    y = data[target]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Decision Tree Classifier
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, predictions)
    print('Accuracy:', accuracy)
    print('Classification Report:\n', classification_report(y_test, predictions))

    return model

# ==============================
# 4. Predict Rainfall for New Data
# ==============================
def predict_new_data(model):
    """Uses the trained model to predict rainfall for new input data."""
    new_data = pd.DataFrame({'Temperature': [25.5], 'Humidity': [70], 'WindSpeed': [10]})
    predicted_rainfall = model.predict(new_data)
    print('Predicted rainfall:', predicted_rainfall)

# ==============================
# 5. Main Execution
# ==============================
def main():
    file_path = 'weather.csv'  # Ensure this file is uploaded to the repo
    data = load_and_preprocess_data(file_path)
    visualize_data(data)
    model = train_and_evaluate_model(data)
    predict_new_data(model)

if __name__ == "__main__":
    main()


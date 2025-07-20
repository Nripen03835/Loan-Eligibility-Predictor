import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class LoanPredictor:
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
        }
        self.best_model = None
        self.preprocessor = None
        self.label_encoder = LabelEncoder()
        
    def load_data(self, train_path, test_path):
        # Load datasets
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        # Separate features and target
        X = train_df.drop(['Loan_ID', 'Loan_Status'], axis=1)
        y = train_df['Loan_Status']
        
        # Encode target variable
        y = self.label_encoder.fit_transform(y)
        
        return X, y, test_df
    
    def preprocess_data(self, X):
        # Identify categorical and numerical columns
        categorical_cols = X.select_dtypes(include=['object']).columns
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
        
        # Preprocessing for numerical data
        numerical_transformer = SimpleImputer(strategy='median')
        
        # Preprocessing for categorical data
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Bundle preprocessing
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])
        
        return self.preprocessor
    
    def train_models(self, X, y):
        # Preprocess the data
        X_processed = self.preprocess_data(X).fit_transform(X)
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42)
        
        best_score = 0
        best_model_name = ''
        
        # Train and evaluate each model
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            
            if score > best_score:
                best_score = score
                best_model_name = name
                self.best_model = model
        
        print(f"Best model: {best_model_name} with accuracy: {best_score:.2f}")
        
        # Generate evaluation metrics
        self.evaluate_model(X_test, y_test, best_model_name)
        
        return self.best_model
    
    def evaluate_model(self, X_test, y_test, model_name):
        # Predict probabilities
        y_probs = self.best_model.predict_proba(X_test)[:, 1]
        
        # ROC Curve
        fpr, tpr, thresholds = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)
        
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                 label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.savefig('static/roc_curve.png')
        plt.close()
        
        # Confusion Matrix
        y_pred = self.best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Rejected', 'Approved'],
                    yticklabels=['Rejected', 'Approved'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig('static/confusion_matrix.png')
        plt.close()
    
    def save_model(self, model_path='models/loan_predictor.pkl'):
        joblib.dump({
            'model': self.best_model,
            'preprocessor': self.preprocessor,
            'label_encoder': self.label_encoder
        }, model_path)
    
    @staticmethod
    def load_model(model_path='models/loan_predictor.pkl'):
        loaded_data = joblib.load(model_path)
        predictor = LoanPredictor()
        predictor.best_model = loaded_data['model']
        predictor.preprocessor = loaded_data['preprocessor']
        predictor.label_encoder = loaded_data['label_encoder']
        return predictor
    
    def prepare_test_data(self, test_df):
        # Drop Loan_ID column if present
        if 'Loan_ID' in test_df.columns:
            test_df = test_df.drop('Loan_ID', axis=1)
        
        # Preprocess the test data
        X_test_processed = self.preprocessor.transform(test_df)
        
        return X_test_processed
    
    def predict(self, input_data):
        # Preprocess the input data
        processed_data = self.preprocessor.transform(input_data)
        
        # Make prediction
        prediction = self.best_model.predict(processed_data)
        proba = self.best_model.predict_proba(processed_data)[:, 1]
        
        # Decode the prediction
        prediction_label = self.label_encoder.inverse_transform(prediction)
        
        return prediction_label[0], proba[0]

def main():
    # Initialize loan predictor
    predictor = LoanPredictor()
    
    # Load data
    X, y, test_df = predictor.load_data('data/Training Dataset.csv', 'data/Test Dataset.csv')
    
    # Train models
    predictor.train_models(X, y)
    
    # Save the model
    predictor.save_model()
    
    print("Model training and evaluation completed!")

if __name__ == '__main__':
    main()
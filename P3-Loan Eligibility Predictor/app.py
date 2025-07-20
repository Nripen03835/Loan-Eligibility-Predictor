from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from loan_predictor import LoanPredictor
import joblib
import os

app = Flask(__name__)

# Load the model when the application starts
model_data = None
if os.path.exists('models/loan_predictor.pkl'):
    model_data = LoanPredictor.load_model()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form data
        form_data = {
            'Gender': request.form.get('gender'),
            'Married': request.form.get('married'),
            'Dependents': request.form.get('dependents'),
            'Education': request.form.get('education'),
            'Self_Employed': request.form.get('self_employed'),
            'ApplicantIncome': float(request.form.get('applicant_income')),
            'CoapplicantIncome': float(request.form.get('coapplicant_income')),
            'LoanAmount': float(request.form.get('loan_amount')),
            'Loan_Amount_Term': float(request.form.get('loan_term')),
            'Credit_History': float(request.form.get('credit_history')),
            'Property_Area': request.form.get('property_area')
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([form_data])
        
        # Use the predict method of the LoanPredictor instance
        prediction_label, probability = model_data.predict(input_df)
        
        # Prepare result
        result = {
            'prediction': prediction_label,
            'probability': round(probability * 100, 2),
            'form_data': form_data
        }
        
        return render_template('results.html', result=result)
    
    return render_template('index.html')

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
    
    if file:
        try:
            # Read the uploaded file
            df = pd.read_csv(file)
            
            # Make sure all required columns are present
            required_columns = [
                'Gender', 'Married', 'Dependents', 'Education', 
                'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 
                'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 
                'Property_Area'
            ]
            
            if not all(col in df.columns for col in required_columns):
                return render_template('index.html', 
                                     error="CSV file doesn't have all required columns")
            
            # Prepare data for prediction
            X_test = df[required_columns]
            
            # Make predictions using the predict method of the LoanPredictor instance
            predictions = []
            probabilities = []
            
            for _, row in X_test.iterrows():
                prediction_label, probability = model_data.predict(pd.DataFrame([row]))
                predictions.append(prediction_label)
                probabilities.append(round(probability * 100, 2))
            
            # Add predictions to the DataFrame
            df['Prediction'] = predictions
            df['Probability'] = probabilities
            
            # Save results to a new CSV file
            results_path = 'static/batch_predictions.csv'
            df.to_csv(results_path, index=False)
            
            return render_template('results.html', 
                                 batch_results=True,
                                 results_file=results_path)
            
        except Exception as e:
            return render_template('index.html', error=str(e))
    
    return redirect(url_for('index'))

if __name__ == '__main__':
    # Train the model if it doesn't exist
    if not os.path.exists('models/loan_predictor.pkl'):
        print("Model not found. Training model...")
        from loan_predictor import LoanPredictor
        predictor = LoanPredictor()
        X, y, test_df = predictor.load_data('data/Training Dataset.csv', 'data/Test Dataset.csv')
        predictor.train_models(X, y)
        predictor.save_model()
    
    app.run(debug=True)
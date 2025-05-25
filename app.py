from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np

model_path = "Loan_approval_model.pkl"

with open(model_path, 'rb') as f:
    model = pickle.load(f)

lr_path = "Loan_determine_model.pkl"  
with open(lr_path, 'rb') as g:
    lr = pickle.load(g)


app = Flask(__name__)
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods = ['POST'])
def loan_approv():
   
    gender = int(request.form['gender'])
    marrital_status = int(request.form['marrital_status'])
    education = int(request.form['education'])
    self_employed = int(request.form['self_employed'])
    applicant_income = float(request.form['applicant_income'])
    coapplicant_income = float(request.form['coapplicant_income'])
    total_income = int(applicant_income) + int(coapplicant_income)
    loan_amount = float(request.form['loan_amount'])
    loan_period = int(request.form['loan_period'])
    credit_history = int(request.form['credit_history'])
    
    features = np.array([[gender, marrital_status, education, self_employed,
                          applicant_income, coapplicant_income, total_income,
                          loan_amount, loan_period, credit_history]])
        
    prediction = model.predict(features)[0]
    predict = lr.predict([[total_income]])
    result = ('Loan Approved', 'Loan Amount: ',int(predict)) if prediction == 0 else 'Loan Rejected'
     
    return f"<h3 style='text-align:center; background-color:#f0f8ff; font-size:24px; color:#333; padding:20px;'>{result}</h3><br><a href='/' style='display:block; text-align:center; font-size:18px; color:blue;'>Back</a>"

if __name__== "__main__":
    app.run(debug=True)
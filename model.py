import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.linear_model import LogisticRegression, LinearRegression 
from sklearn.metrics import mean_absolute_error, r2_score, confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split

np.random.seed(42)

n_loanapplicant = 500

loan_id = np.random.randint(10001, 20001, n_loanapplicant)
gender = np.random.randint(0, 2, n_loanapplicant) # 0 : female, 1 : male
marrital_status = np.random.randint(0, 2, n_loanapplicant) # 0 : married, 1 : single
education = np.random.randint(0, 2, n_loanapplicant) # 0: literate, 1 : illiterate
self_employed = np.random.randint(0, 2, n_loanapplicant) # 0 : self-employed, 1 : service
applicant_income = np.random.normal(50, 15, n_loanapplicant) 
coapplicant_income = np.random.normal(50, 15, n_loanapplicant)
total_income = applicant_income + coapplicant_income
loan_amount = np.random.randint(100000,500000, n_loanapplicant)
loan_period = np.random.randint(1, 4, n_loanapplicant)
credit_history = np.random.randint(0, 2, n_loanapplicant) # 0 : good, 1 : bad

credit_score = ((total_income < 70) & (credit_history != 0)).astype(int) # 0 : good, 1 : bad

df = pd.DataFrame({
    "Loan_ID" : loan_id,
    "Gender " : gender,
    "Marrital_Status" : marrital_status,
    "Education " : education,
    "Self_Employed":self_employed,
    "Applicant_Income": applicant_income,
    "Co-Applicant_Income":coapplicant_income,
    "Total_Amount": total_income,
    "Loan_Amount":loan_amount,
    "Loan_Period": loan_period,
    "Credit_history ": credit_history,
    "Credit_Score":credit_score
})

df.to_csv("Loan_approval_dataset.csv", index=False)
print(df.head())

def loan():
    df2 = pd.read_csv("Loan_approval_dataset.csv")
    print(df2.isnull().sum())

#plt.figure(figsize=(20,10))
#plt.tight_layout
#sns.heatmap(data=df2.corr(), cmap='crest', annot = True)
#plt.show()

    X = df2.drop(columns=["Credit_Score","Loan_ID"])
    y = df2['Credit_Score']
    print(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("Accuracy Score: ", acc)

    print("Confusion Matrix", confusion_matrix(y_test, y_pred))
    
    
    x = df2['Total_Amount']
    Y = df2['Loan_Amount']
    
    x1 = sorted(x)
    y1 = sorted(Y)
    
    x2 = np.array(x1)
    y2 = np.array(y1)
    
    x2 = x2.reshape(-1, 1)
    y2 = y2.reshape(-1, 1)
    
    lr = LinearRegression()
    lr.fit(x2, y2)

    import pickle
    model_path = "Loan_approval_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print("Model saved")
    
    lr_path = "Loan_determine_model.pkl"
    with open(lr_path, 'wb') as g:
        pickle.dump(lr, g)
    
    print("Loan Determination model saved")

if __name__ == "__main__":
    loan()
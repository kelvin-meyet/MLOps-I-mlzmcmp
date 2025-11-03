# --- Load the saved model
import pickle

print("loading trained model...")

model_file = "model_C=0.5.bin"

with open(model_file, "rb") as f_in:
    dv, model = pickle.load(f_in)


print("loading customer details")

customer = {
    "gender": "female",
    "seniorcitizen": 0,
    "partner": "yes",
    "dependents": "no",
    "phoneservice": "no",
    "multiplelines": "no_phone_service",
    "internetservice": "dsl",
    "onlinesecurity": "no",
    "onlinebackup": "yes",
    "deviceprotection": "no",
    "techsupport": "no",
    "streamingtv": "no",
    "streamingmovies": "no",
    "contract": "month-to-month",
    "paperlessbilling": "yes",
    "paymentmethod": "electronic_check",
    "tenure": 1,
    "monthlycharges": 29.85,
    "totalcharges": 29.85,
}

X = dv.transform([customer])
churn_prediction = model.predict_proba(X)[0, 1]
print("")

print("input", customer)

print()

print("churn probability", churn_prediction)

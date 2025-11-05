import requests

url = "http://localhost:9696/predict"


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
    #"age": 25  
}




response = requests.post(url, json=customer)
predictions = response.json()

print('response:', predictions)

if predictions['churn'] >= 0.5:
    print('customer likely to churn, send email with promo')
else:
    print('Customer is not likely to churn, Do not send promo')





# ---> For prediction service 

print("Loading saved model binaries for prediction ...")
#load pipeline model
import pickle
model_file = 'pipeline_model_C=0.5.bin'


with open(model_file, 'rb') as f_in:
    training_pipeline = pickle.load(f_in) 

print(training_pipeline)


print('Predicting the churn prob of a random customer ...')
#predict a customer with pipeline

test_customer2 = {
    'gender': 'male',
    'seniorcitizen': 0,
    'partner': 'no',
    'dependents': 'yes',
    'phoneservice': 'no',
    'multiplelines': 'no_phone_service',
    'internetservice': 'dsl',
    'onlinesecurity': 'no',
    'onlinebackup': 'yes',
    'deviceprotection': 'no',
    'techsupport': 'no',
    'streamingtv': 'no',
    'streamingmovies': 'no',
    'contract': 'month-to-month',
    'paperlessbiling': 'yes',
    'paymentmethod': 'electronic_check',
    'tenure': 1,
    'monthlycharges': 29.85,
    'totalcharges': (1 * 29.85)
}

churn_proba = training_pipeline.predict_proba(test_customer2)[0,1]
print('The churn probability of a random customer is', churn_proba.round(2))

print('')
print("Next line of action is: ")

if churn_proba >= 0.5:
    print('send email with promo')
else:
    print('Do not send promo')



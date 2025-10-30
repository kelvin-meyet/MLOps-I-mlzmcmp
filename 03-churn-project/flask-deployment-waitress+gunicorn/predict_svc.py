# --- Load the model
import pickle
from flask import Flask
from flask import request
from flask import jsonify


# load the model file
model_file = "model_C=0.5.bin"

with open(model_file, "rb") as f_in:
    dv, model = pickle.load(f_in)


# start creating flask app
app = Flask("churn")  # name of flask app


@app.route(
    "/predict-churn", methods=["POST"]
)  # decorator - when someone sends a POST request to /predict-churn, run the function below
def predict():
    customer = request.get_json()
    # flask can only accept customer data as json, and we are extracting the body of the request as json

    X = dv.transform([customer])
    churn_prediction = model.predict_proba(X)[0, 1]
    churn = churn_prediction >= 0.5

    # response from flask will also be json
    result = {"churn_probability": float(churn_prediction), "churn": bool(churn)}

    return jsonify(result)


# this run the file directly and not when imported

# every python file has inbuilt variable called __name__, when you run any file directly (ping_script.py) python sets __name__ to "__main__"
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)

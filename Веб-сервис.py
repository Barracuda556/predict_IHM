from flask import Flask, request, jsonify
import numpy as np
import joblib
import requests
import io

url = "https://github.com/Barracuda556/predict_IHM/raw/refs/heads/main/catboost_IHM.pkl"

response = requests.get(url)
response.raise_for_status()
model = joblib.load(io.BytesIO(response.content))

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([data['features']])  # Предполагаем, что приходит {"features": [...]}
    probability = model.predict_proba(features)[:, 1]#[0, 1]  # Для бинарной классификации
    return jsonify({'probability': float(probability[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


# Пример запроса
# curl -X POST -H "Content-Type: application/json" -d "{\"features\": [81.0, 75.0, 185.0, 89.0, 0.0, 0.0, 111.0, 0.0, 0.0, 0.0, 7.0, 0.0]}"  http://localhost:5000/predict

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import requests
import io

# Базовый URL репозитория
base_url = "https://github.com/Barracuda556/predict_IHM/raw/main/"

# Список имен файлов
model_files = [
    'model_1.pkl', 'model_2.pkl', 'model_3.pkl', 'model_4.pkl', 'model_5.pkl',
    'scaler_1.pkl', 'scaler_2.pkl', 'scaler_3.pkl', 'scaler_4.pkl', 'scaler_5.pkl'
]

# Загружаем все файлы
loaded_objects = {}
for file in model_files:
    try:
        url = base_url + file
        response = requests.get(url)
        response.raise_for_status()
        loaded_objects[file.split('.')[0]] = joblib.load(io.BytesIO(response.content))
        print(f"Успешно загружен {file}")
    except Exception as e:
        print(f"Ошибка при загрузке {file}: {str(e)}")

model_1 = loaded_objects['model_1']
model_2 = loaded_objects['model_2']
model_3 = loaded_objects['model_3']
model_4 = loaded_objects['model_4']
model_5 = loaded_objects['model_5']
scaler_1 = loaded_objects['scaler_1']
scaler_2 = loaded_objects['scaler_2']
scaler_3 = loaded_objects['scaler_3']
scaler_4 = loaded_objects['scaler_4']
scaler_5 = loaded_objects['scaler_5']

threshold_1 = 0.08
threshold_2 = 0.087
threshold_3 = 0.098
threshold_4 = 0.101
threshold_5 = 0.11

app = Flask(__name__)

CORS(app, resources={
    r"/predict": {
        "origins": "*",
        "methods": ["OPTIONS", "POST"],
        "allow_headers": ["Content-Type"]
    }
})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if len(data['features']) == 11:
        if data['features'][5] == None: 
            del data['features'][5]
            model = model_1
            scaler = scaler_1
            threshold = threshold_1
        else: 
            model = model_2
            scaler = scaler_2
            threshold = threshold_2
    elif len(data['features']) == 13: 
        model = model_3
        scaler = scaler_3
        threshold = threshold_3
    elif len(data['features']) == 14: 
        model = model_4
        scaler = scaler_4
        threshold = threshold_4
    elif len(data['features']) == 15: 
        model = model_5
        scaler = scaler_5
        threshold = threshold_5
    features = np.array([data['features']]) 
    data = scaler.transform(features)
    probability = model.predict_proba(data)[:, 1]
    if probability >= threshold: risk = 'Высокий риск'
    elif probability >= threshold/2: risk = 'Средний риск'
    else: risk = 'Низкий риск'
    return jsonify({'probability': float(probability[0]), 'risk': risk})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

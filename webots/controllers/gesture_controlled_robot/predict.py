import joblib
import numpy as np

def load_model(model_name):
    model = joblib.load(f'{model_name}_model.pkl')
    scaler = joblib.load(f'{model_name}_scaler.pkl')
    return model, scaler

def predict_new_gesture(new_landmarks):
    # Load the saved KNN model and scaler
    model, scaler = load_model('knn')
    
    # Reshape the landmarks into the correct format
    new_data = np.array(new_landmarks).reshape(1, -1)
    
    # Scale the data and predict
    new_data_scaled = scaler.transform(new_data)
    prediction = model.predict(new_data_scaled)
    return prediction[0]

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
import category_encoders as ce
import uvicorn

app = FastAPI(title="Student Dropout Prediction API")

model = pickle.load(open("best_xgb_model.pkl", "rb"))
feature_columns = pickle.load(open("feature_columns.pkl", "rb"))
label_encoder = pickle.load(open("target_encoder.pkl", "rb"))

invalid_cols = ["Dropout_Probability", "Predicted_Dropout", "Actual_Dropout", "special_needs_flag"]
feature_columns = [col for col in feature_columns if col not in invalid_cols]

class StudentData(BaseModel):
    features: dict

@app.post("/predict")
def predict(data: StudentData):
    try:
        feature_vector = [float(data.features[col]) for col in feature_columns]
        
        if len(feature_vector) != len(feature_columns):
            return {"error": f"Feature shape mismatch: expected {len(feature_columns)}, got {len(feature_vector)}"}

        feature_array = np.array([feature_vector])

        prediction = int(model.predict(feature_array)[0])
        probability = float(model.predict_proba(feature_array)[0][1])

        label_map = {0: "Not Dropout", 1: "Dropout"}
        prediction_label = label_map[prediction]

        return {
            "prediction": prediction_label,
            "dropout_probability": probability
        }

    except KeyError as e:
        return {"error": f"Missing feature: {e}"}
    except Exception as e:
        return {"error": str(e)}

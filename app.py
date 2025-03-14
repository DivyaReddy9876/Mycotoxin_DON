from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import numpy as np
from werkzeug.serving import run_simple

# Define the model
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(3, 1)

    def forward(self, x):
        return self.fc(x)

# Load the model
model = MyModel()
model.load_state_dict(torch.load('model.pth', map_location='cpu'))
model.eval()

# Flask App
app = Flask(__name__)

# Home Page - Retrieve Model and Input Details
@app.route("/", methods=["GET"])
def home():
    input_shape = model.fc.in_features  # Get input shape dynamically
    output_shape = model.fc.out_features  # Get output shape
    return jsonify({
        "model_name": model.__class__.__name__,
        "input_shape": input_shape,
        "output_shape": output_shape,
        "api_endpoints": ["/predict (POST)"]
    })

# Prediction Endpoint (POST Request)
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json["input"]
        tensor_input = torch.tensor([data], dtype=torch.float32)

        with torch.no_grad():
            prediction = model(tensor_input).item()

        return jsonify({"prediction": prediction})
    
    except Exception as e:
        return jsonify({"error": str(e)})

# Start Flask in Jupyter (Optimized Execution)
run_simple('0.0.0.0', 5000, app, use_reloader=False, use_debugger=False)

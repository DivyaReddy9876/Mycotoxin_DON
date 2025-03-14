# Mycotoxin_DON
---

# **Mycotoxin Prediction Using Machine Learning**  

## **Project Overview**  
This project aims to develop a **machine learning-based system** to predict **DON (Deoxynivalenol) mycotoxin concentration** in corn samples using **hyperspectral imaging data**. The pipeline includes:  
✔ Data Preprocessing (handling missing values, normalization, PCA)  
✔ Feature Engineering and Selection  
✔ Model Training, Evaluation, and Optimization  
✔ Flask API Deployment for Real-Time Prediction  

This system is designed for **modular, interpretable, and production-ready deployment** in real-world agricultural applications.  

---

## **1. Setup Instructions**  

### **1.1 Prerequisites**  
Before running this project, ensure you have:  
- **Python 3.12.2** installed  
- **Jupyter Notebook** for interactive development  
- **Flask** for API deployment  

---

### **1.2 Install Dependencies**  
To install all required packages, run:  
```bash  
pip install -r requirements.txt  
```  
If `requirements.txt` is missing, install dependencies manually:  
```bash  
pip install numpy pandas scikit-learn torch flask werkzeug matplotlib seaborn joblib  
```  

---

### **1.3 Download or Clone Repository**  
Clone this repository using Git:  
```bash  
git clone https://github.com/yourusername/mycotoxin-prediction.git  
cd mycotoxin-prediction  
```  

---

## **2. Running the Code**  

### **2.1 Data Processing & Model Training**  
Run the **Jupyter Notebook** (`notebook.ipynb`) step by step:  
✔ Load dataset (`data/mycotoxin_data.csv`)  
✔ Handle missing values and outliers  
✔ Apply **StandardScaler** and **PCA**  
✔ Train models (Linear Regression, Random Forest)  
✔ Save the best-trained model (`model.pth`)  

---

### **2.2 Running the Flask API**  

#### **Option 1: Running in Jupyter Notebook**  
Since **Jupyter Notebook** does not support `app.run()`, use:  
```python  
from werkzeug.serving import run_simple  
run_simple('0.0.0.0', 5000, app, use_reloader=False, use_debugger=False)  
```  

#### **Option 2: Running in Terminal**  
Execute this in your command line:  
```bash  
python app.py  
```  

✔ Open **http://192.168.1.11:5000** in a web browser  
✔ Use **POST /predict** to make predictions  

---

## **3. Repository Structure**  

📂 **data/** → Contains raw and processed datasets  
📂 **models/** → Stores trained machine learning models  
📂 **notebooks/** → Jupyter notebooks for training & analysis  
📂 **api/** → Flask API scripts for model deployment  
📂 **utils/** → Helper functions (preprocessing, feature engineering)  
📄 **requirements.txt** → Required dependencies  
📄 **app.py** → Flask API server  
📄 **notebook.ipynb** → Machine learning training pipeline  
📄 **README.md** → Project documentation  

---

## **4. API Endpoints**  

### **4.1 Home Endpoint**  
📌 **GET /**  
**Description:** Retrieves model details and available API endpoints.  

**Response:**  
```json  
{  
  "model_name": "MyModel",  
  "input_shape": 3,  
  "output_shape": 1,  
  "api_endpoints": ["/predict (POST)"]  
}  
```  

---

### **4.2 Prediction Endpoint**  
📌 **POST /predict**  
**Description:** Accepts an input array and returns the predicted DON concentration.  

#### **Example Request:**  
```json  
{ "input": [0.12, 0.45, 0.78] }  
```  

#### **Example Response:**  
```json  
{ "prediction": 2.54 }  
```  

✔ Ensure data is sent as **JSON format** in the request body.  

---

## **5. Model Training & Selection**  

### **5.1 Train-Test Split**  
✔ Used **80% Training / 20% Testing** with `train_test_split()` from Scikit-Learn  
✔ Implemented **5-fold cross-validation** for robust performance  

---

### **5.2 Model Selection**  

#### **Baseline Model**  
✔ Started with **Linear Regression** as a benchmark  

#### **Final Model: Random Forest Regressor**  
✔ Chosen for its ability to handle **high-dimensional, nonlinear data**  

✔ **Hyperparameter Tuning:**  
  - `n_estimators`: Number of trees  
  - `max_depth`: Tree depth limit  
  - `min_samples_split`: Minimum samples to split nodes  

✔ **Achieved R² Score:** **0.85**  

---

## **6. Model Evaluation**  

### **6.1 Metrics Used**  
✔ **Mean Absolute Error (MAE):** Measures average error magnitude  
✔ **Mean Squared Error (MSE):** Penalizes larger errors  
✔ **R² Score:** Measures model accuracy  

---

### **6.2 Visual Evaluation**  
✔ **Scatter Plot:** Actual vs. Predicted DON concentration  
✔ **Residual Analysis:** Ensures errors are randomly distributed  

---

## **7. Deployment Considerations**  

### **7.1 Running Flask on a Server**  
To deploy on an external server, modify **app.py**:  
```python  
app.run(host='0.0.0.0', port=5000)  
```  

To keep the Flask app running:  
```bash  
nohup python app.py &  
```  

---

## **8. Key Findings & Improvements**  

### **8.1 Findings**  
✔ **PCA reduced training time** without losing accuracy  
✔ **Random Forest performed better** than Linear Regression with an **R² score of 0.85**  
✔ **Certain wavelengths were highly predictive** of DON concentration  

---

### **8.2 Suggested Improvements**  
✅ Implement **Neural Networks (e.g., CNN, LSTM)**  
✅ Explore **Ensemble Models (e.g., XGBoost, LightGBM)**  
✅ Use **advanced feature selection techniques**  
✅ Optimize the model for **real-time predictions**  

---

## **9. Frequently Asked Questions (FAQ)**  

### **Q1: How do I send a request to the API?**  
Use **Postman** or **cURL**:  
```bash  
curl -X POST http://192.168.1.11:5000/predict -H "Content-Type: application/json" -d '{"input": [0.12, 0.45, 0.78]}'  
```  

### **Q2: Can I train the model on my own dataset?**  
Yes! Replace `data/mycotoxin_data.csv` with your dataset and retrain using `notebook.ipynb`.  

### **Q3: How do I update dependencies?**  
Run:  
```bash  
pip install --upgrade -r requirements.txt  
```  

---

## **10. Additional Resources**  

✔ **Scikit-Learn Documentation:** https://scikit-learn.org/stable/  
✔ **Flask Documentation:** https://flask.palletsprojects.com/  
✔ **PyTorch Documentation:** https://pytorch.org/  

---

---


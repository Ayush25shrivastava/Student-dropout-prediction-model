# Student-dropout-prediction-model
This project is made under Data Vision event , Avishkar 2025, by team Tensor. 
This project predicts student dropout risk using a trained XGBoost model.
It provides:

🧠 Machine Learning API using FastAPI

🖥 Interactive Web UI using Streamlit

🔄 Real-time prediction for individual students

📊 Risk-based dropout probability

### ⚙️ Installation & Setup
1️⃣ Create & activate virtual environment

Windows:

python -m venv venv
venv\Scripts\activate

macOS/Linux:

python3 -m venv venv
source venv/bin/activate

### Install required packages
pip install -r requirements.txt

### 🚀 Run the Application

### Start Backend (API Server)
cd backend
uvicorn main:app --reload --port 9000   

API runs at: http://localhost:9000

### Start Frontend (Web UI)

Open a new terminal:

cd frontend
streamlit run app.py

Frontend URL will open automatically in browser:
📍 http://localhost:8501
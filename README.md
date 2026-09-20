# Loan Approval & Risk Assessment Dashboard

Automated credit risk evaluation model using a FastAPI backend and a React (Vite) frontend.

## Architecture

This project has been upgraded from a Streamlit script into a full-stack application to provide a more robust and scalable architecture:
- **Frontend (`/frontend`)**: React application bootstrapped with Vite, styled with Tailwind CSS, and using Recharts for data visualization. Matches the exact dark-themed design system.
- **Backend (`/backend`)**: FastAPI server that serves the trained Random Forest model and exposes endpoints for predictions and model performance metrics.
- **Machine Learning**: Scikit-Learn pipeline (`loan_model.py`) that handles data preprocessing, training multiple models, hyperparameter tuning, and serializing the best model.

## Setup & Running

### 1. Backend (FastAPI)
The backend runs the machine learning model and serves the API.

```bash
# Navigate to the backend directory
cd backend

# Install dependencies
pip install -r requirements.txt

# Run the server (starts on http://localhost:8000)
python -m uvicorn main:app --reload
```

### 2. Frontend (React)
The frontend serves the user interface.

```bash
# Navigate to the frontend directory
cd frontend

# Install dependencies
npm install

# Run the development server (starts on http://localhost:3000)
npm run dev -- --port 3000
```

## Features
- **Applicant Assessment**: Interactive form to input applicant details and receive an immediate risk assessment.
- **Model Performance**: Visualizations of the model's accuracy, feature importances, and ROC curve.
- **Documentation**: Detailed explanation of the dataset, methodology, and API reference.

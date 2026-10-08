import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "backend"))
import json
import uuid
from datetime import datetime, timedelta
import random
from app import run_prediction, ApplicantData

random.seed(42)

NAMES = [
    "Vikram Malhotra", "Ananya Sharma", "Rajesh Iyer", "Priya Nair",
    "Arjun Patel", "Sneha Rao", "Rohan Gupta", "Deepika Joshi",
    "Karthik Menon", "Meera Kulkarni", "Aditya Verma", "Tanvi Bhatia",
    "Siddharth Das", "Neha Kapoor", "Varun Saxena", "Pooja Reddy",
    "Alok Kumar", "Divya Pillai", "Manish Choudhary", "Ritu Singhania",
    "Amit Deshmukh", "Swati Nambiar", "Nikhil Chopra", "Sunita Agarwal"
]

SCENARIOS = [
    # Strong profiles (Approved, Low risk)
    {"gender": "Male", "married": "Yes", "dependents": "1", "education": "Graduate", "self_employed": "No", "inc": 8500, "coinc": 3200, "loan": 140, "term": 360, "cibil": 782, "prop": "Semiurban", "res": 4500000, "comm": 1200000, "bank": 1800000, "lux": 600000},
    {"gender": "Female", "married": "No", "dependents": "0", "education": "Graduate", "self_employed": "No", "inc": 7200, "coinc": 0, "loan": 110, "term": 360, "cibil": 810, "prop": "Urban", "res": 3200000, "comm": 0, "bank": 1500000, "lux": 400000},
    {"gender": "Male", "married": "Yes", "dependents": "2", "education": "Graduate", "self_employed": "No", "inc": 9500, "coinc": 4000, "loan": 220, "term": 360, "cibil": 775, "prop": "Urban", "res": 6500000, "comm": 2500000, "bank": 2200000, "lux": 1100000},
    {"gender": "Female", "married": "Yes", "dependents": "1", "education": "Graduate", "self_employed": "No", "inc": 6800, "coinc": 2400, "loan": 130, "term": 240, "cibil": 790, "prop": "Semiurban", "res": 3800000, "comm": 0, "bank": 1400000, "lux": 300000},
    {"gender": "Male", "married": "No", "dependents": "0", "education": "Graduate", "self_employed": "No", "inc": 6000, "coinc": 0, "loan": 95, "term": 180, "cibil": 760, "prop": "Urban", "res": 2800000, "comm": 0, "bank": 1200000, "lux": 250000},
    {"gender": "Female", "married": "Yes", "dependents": "0", "education": "Graduate", "self_employed": "No", "inc": 8200, "coinc": 3500, "loan": 160, "term": 360, "cibil": 825, "prop": "Semiurban", "res": 5200000, "comm": 1500000, "bank": 2100000, "lux": 800000},
    {"gender": "Male", "married": "Yes", "dependents": "2", "education": "Graduate", "self_employed": "Yes", "inc": 11000, "coinc": 2000, "loan": 190, "term": 360, "cibil": 755, "prop": "Urban", "res": 7000000, "comm": 4000000, "bank": 3000000, "lux": 1200000},
    {"gender": "Female", "married": "No", "dependents": "0", "education": "Graduate", "self_employed": "No", "inc": 5800, "coinc": 0, "loan": 85, "term": 360, "cibil": 745, "prop": "Semiurban", "res": 2500000, "comm": 0, "bank": 950000, "lux": 200000},
    {"gender": "Male", "married": "Yes", "dependents": "1", "education": "Graduate", "self_employed": "No", "inc": 7500, "coinc": 2800, "loan": 150, "term": 360, "cibil": 768, "prop": "Urban", "res": 4200000, "comm": 800000, "bank": 1600000, "lux": 500000},
    {"gender": "Female", "married": "Yes", "dependents": "2", "education": "Graduate", "self_employed": "No", "inc": 8900, "coinc": 3100, "loan": 175, "term": 360, "cibil": 805, "prop": "Semiurban", "res": 5800000, "comm": 1800000, "bank": 2400000, "lux": 900000},
    {"gender": "Male", "married": "No", "dependents": "0", "education": "Graduate", "self_employed": "No", "inc": 6400, "coinc": 0, "loan": 105, "term": 360, "cibil": 750, "prop": "Urban", "res": 3100000, "comm": 0, "bank": 1300000, "lux": 350000},
    {"gender": "Male", "married": "Yes", "dependents": "1", "education": "Graduate", "self_employed": "No", "inc": 9200, "coinc": 3800, "loan": 210, "term": 360, "cibil": 788, "prop": "Semiurban", "res": 6200000, "comm": 2100000, "bank": 2500000, "lux": 1000000},

    # Moderate profiles (Approved with review / borderline)
    {"gender": "Male", "married": "No", "dependents": "1", "education": "Not Graduate", "self_employed": "No", "inc": 4200, "coinc": 0, "loan": 120, "term": 360, "cibil": 690, "prop": "Rural", "res": 1800000, "comm": 0, "bank": 600000, "lux": 100000},
    {"gender": "Female", "married": "Yes", "dependents": "2", "education": "Graduate", "self_employed": "Yes", "inc": 5400, "coinc": 1500, "loan": 160, "term": 360, "cibil": 680, "prop": "Urban", "res": 3000000, "comm": 800000, "bank": 900000, "lux": 250000},
    {"gender": "Male", "married": "Yes", "dependents": "3+", "education": "Graduate", "self_employed": "No", "inc": 6200, "coinc": 1800, "loan": 180, "term": 240, "cibil": 695, "prop": "Semiurban", "res": 3500000, "comm": 0, "bank": 1100000, "lux": 200000},
    {"gender": "Female", "married": "No", "dependents": "0", "education": "Not Graduate", "self_employed": "No", "inc": 4600, "coinc": 0, "loan": 110, "term": 360, "cibil": 675, "prop": "Semiurban", "res": 1900000, "comm": 0, "bank": 700000, "lux": 150000},

    # High risk / Rejected profiles (Poor credit, high DTI, lower income)
    {"gender": "Male", "married": "No", "dependents": "2", "education": "Not Graduate", "self_employed": "Yes", "inc": 2400, "coinc": 0, "loan": 210, "term": 360, "cibil": 580, "prop": "Rural", "res": 1200000, "comm": 0, "bank": 250000, "lux": 50000},
    {"gender": "Female", "married": "No", "dependents": "1", "education": "Graduate", "self_employed": "No", "inc": 2800, "coinc": 0, "loan": 190, "term": 360, "cibil": 540, "prop": "Rural", "res": 1400000, "comm": 0, "bank": 300000, "lux": 50000},
    {"gender": "Male", "married": "Yes", "dependents": "3+", "education": "Not Graduate", "self_employed": "Yes", "inc": 3100, "coinc": 800, "loan": 240, "term": 360, "cibil": 595, "prop": "Rural", "res": 1600000, "comm": 0, "bank": 400000, "lux": 80000},
    {"gender": "Male", "married": "No", "dependents": "0", "education": "Not Graduate", "self_employed": "No", "inc": 2200, "coinc": 0, "loan": 160, "term": 180, "cibil": 510, "prop": "Urban", "res": 900000, "comm": 0, "bank": 180000, "lux": 0},
    {"gender": "Female", "married": "Yes", "dependents": "3+", "education": "Not Graduate", "self_employed": "Yes", "inc": 2900, "coinc": 500, "loan": 230, "term": 360, "cibil": 565, "prop": "Rural", "res": 1300000, "comm": 0, "bank": 320000, "lux": 60000},
    {"gender": "Male", "married": "No", "dependents": "1", "education": "Graduate", "self_employed": "Yes", "inc": 3500, "coinc": 0, "loan": 260, "term": 360, "cibil": 530, "prop": "Semiurban", "res": 1700000, "comm": 0, "bank": 390000, "lux": 70000},
    {"gender": "Female", "married": "No", "dependents": "0", "education": "Not Graduate", "self_employed": "No", "inc": 2100, "coinc": 0, "loan": 140, "term": 360, "cibil": 520, "prop": "Rural", "res": 850000, "comm": 0, "bank": 150000, "lux": 0},
    {"gender": "Male", "married": "Yes", "dependents": "2", "education": "Not Graduate", "self_employed": "Yes", "inc": 3300, "coinc": 700, "loan": 220, "term": 360, "cibil": 590, "prop": "Rural", "res": 1500000, "comm": 0, "bank": 360000, "lux": 50000},
]

records = []
base_time = datetime(2026, 10, 8, 14, 30)

for i, (name, s) in enumerate(zip(NAMES, SCENARIOS)):
    app_data = ApplicantData(
        gender=s["gender"],
        married=s["married"],
        dependents=s["dependents"],
        education=s["education"],
        self_employed=s["self_employed"],
        applicant_income=float(s["inc"]),
        coapplicant_income=float(s["coinc"]),
        loan_amount=float(s["loan"]),
        loan_amount_term=int(s["term"]),
        credit_history=1 if s["cibil"] >= 650 else 0,
        property_area=s["prop"],
        cibil_score=s["cibil"],
        annual_income=float((s["inc"] + s["coinc"]) * 12),
        residential_assets=float(s["res"]),
        commercial_assets=float(s["comm"]),
        bank_assets=float(s["bank"]),
        luxury_assets=float(s["lux"])
    )
    result = run_prediction(app_data)
    created_at = (base_time - timedelta(days=random.randint(0, 18), hours=random.randint(1, 10))).isoformat()
    
    rec = {
        "id": str(uuid.uuid4()),
        "ref_code": f"LG-{100 + i + 1:06d}",
        "user_email": "analyst@loanguard.io",
        "applicant_name": name,
        "applicant": app_data.dict(),
        "result": result,
        "created_at": created_at
    }
    records.append(rec)

# Sort records descending by created_at
records.sort(key=lambda r: r["created_at"], reverse=True)

apps_file = Path(__file__).parent.parent / "backend" / "data" / "applications.json"
with open(apps_file, "w") as f:
    json.dump(records, f, indent=2)

print(f"Generated {len(records)} realistic seed applications in {apps_file}")

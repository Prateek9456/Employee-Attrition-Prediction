from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

print("Loading models...")
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')
explainer = joblib.load('explainer.pkl')
label_encoders = joblib.load('label_encoders.pkl')
print("✓ All models loaded successfully!")

app = FastAPI(title="Employee Attrition Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class EmployeeInput(BaseModel):
    Age: int
    BusinessTravel: str
    DailyRate: int
    Department: str
    DistanceFromHome: int
    Education: int
    EducationField: str
    EnvironmentSatisfaction: int
    Gender: str
    HourlyRate: int
    JobInvolvement: int
    JobLevel: int
    JobRole: str
    JobSatisfaction: int
    MaritalStatus: str
    MonthlyIncome: int
    MonthlyRate: int
    NumCompaniesWorked: int
    OverTime: str
    PercentSalaryHike: int
    PerformanceRating: int
    RelationshipSatisfaction: int
    StockOptionLevel: int
    TotalWorkingYears: int
    TrainingTimesLastYear: int
    WorkLifeBalance: int
    YearsAtCompany: int
    YearsInCurrentRole: int
    YearsSinceLastPromotion: int
    YearsWithCurrManager: int

class PredictionOutput(BaseModel):
    risk_probability: float
    risk_level: str
    top_factors: List[Dict]
    recommended_actions: List[Dict]

@app.get("/")
async def root():
    return {
        "status": "Attrition Prediction API is running",
        "version": "1.0",
        "endpoints": ["/predict", "/docs"]
    }

@app.post("/predict", response_model=PredictionOutput)
async def predict_attrition(employee: EmployeeInput):
    try:
        # Convert to dataframe
        data = pd.DataFrame([employee.dict()])
        
        # Encode categorical variables
        categorical_cols = ['BusinessTravel', 'Department', 'EducationField', 
                           'Gender', 'JobRole', 'MaritalStatus', 'OverTime']
        for col in categorical_cols:
            if col in data.columns and col in label_encoders:
                try:
                    data[col] = label_encoders[col].transform(data[col])
                except:
                    data[col] = 0
        
        # Feature engineering
        data['PromotionGap'] = data['YearsAtCompany'] - data['YearsSinceLastPromotion']
        data['SatisfactionWorkloadRatio'] = data['JobSatisfaction'] / (data['DistanceFromHome'] + 1)
        data['CareerStagnation'] = ((data['YearsSinceLastPromotion'] > 3) & 
                                     (data['PerformanceRating'] >= 3)).astype(int)
        data['IncomeToAgeRatio'] = data['MonthlyIncome'] / data['Age']
        data['ExperienceToPromotionRatio'] = data['YearsAtCompany'] / (data['YearsSinceLastPromotion'] + 1)
        
        # Ensure correct column order
        data = data[feature_names]
        
        # Scale
        data_scaled = scaler.transform(data)
        
        # Predict
        probability = float(model.predict_proba(data_scaled)[0][1])
        
        # Risk level
        if probability > 0.7:
            risk_level = "HIGH"
        elif probability > 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        # SHAP Explanations
        shap_values = explainer.shap_values(data_scaled)
        
        # Top factors
        feature_impact = list(zip(feature_names, shap_values[0]))
        feature_impact.sort(key=lambda x: abs(x[1]), reverse=True)
        
        top_factors = []
        for feat, impact in feature_impact[:5]:
            if abs(impact) > 0.01:
                top_factors.append({
                    "factor": feat,
                    "impact": float(impact),
                    "direction": "increases risk" if impact > 0 else "decreases risk",
                    "value": float(data[feat].iloc[0])
                })
        
        # Generate actions
        actions = generate_actions(top_factors, probability)
        
        return PredictionOutput(
            risk_probability=probability,
            risk_level=risk_level,
            top_factors=top_factors,
            recommended_actions=actions
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def generate_actions(factors, risk_prob):
    actions = []
    
    action_map = {
        'YearsSinceLastPromotion': {
            'action': 'Schedule promotion review meeting',
            'priority': 'Immediate',
            'description': 'Employee overdue for career advancement'
        },
        'JobSatisfaction': {
            'action': 'Conduct 1-on-1 satisfaction assessment',
            'priority': 'Immediate',
            'description': 'Address job satisfaction concerns with manager'
        },
        'MonthlyIncome': {
            'action': 'Review compensation against market benchmarks',
            'priority': 'High',
            'description': 'Potential salary adjustment needed'
        },
        'WorkLifeBalance': {
            'action': 'Implement flexible work arrangements',
            'priority': 'Medium',
            'description': 'Improve work-life balance options'
        },
        'OverTime': {
            'action': 'Review workload and redistribute tasks',
            'priority': 'High',
            'description': 'Excessive overtime detected'
        },
        'DistanceFromHome': {
            'action': 'Discuss remote work or relocation assistance',
            'priority': 'Medium',
            'description': 'Long commute may be affecting retention'
        },
        'PromotionGap': {
            'action': 'Create career development plan',
            'priority': 'Immediate',
            'description': 'Large gap between tenure and promotions'
        },
        'EnvironmentSatisfaction': {
            'action': 'Improve workplace environment and culture',
            'priority': 'Medium',
            'description': 'Employee dissatisfied with work environment'
        },
        'YearsWithCurrManager': {
            'action': 'Consider manager change or coaching',
            'priority': 'High',
            'description': 'Potential manager relationship issues'
        }
    }
    
    for factor in factors:
        if factor['impact'] > 0:
            factor_name = factor['factor']
            for key in action_map:
                if key in factor_name:
                    actions.append(action_map[key])
                    break
    
    if risk_prob > 0.7 and len(actions) > 0:
        actions.insert(0, {
            'action': 'URGENT: Schedule immediate retention discussion',
            'priority': 'Critical',
            'description': 'High flight risk - immediate intervention required'
        })
    
    if len(actions) == 0:
        actions.append({
            'action': 'Continue regular check-ins',
            'priority': 'Low',
            'description': 'Employee showing good retention indicators'
        })
    
    return actions

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*50)
    print("🚀 Starting API Server...")
    print("="*50)
    print("📍 API will be available at: http://localhost:8000")
    print("📚 Documentation at: http://localhost:8000/docs")
    print("\nPress CTRL+C to stop the server")
    print("="*50 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
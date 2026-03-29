import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
import shap
import warnings
warnings.filterwarnings('ignore')

print("=" * 50)
print("EMPLOYEE ATTRITION MODEL TRAINING")
print("=" * 50)

# STEP 1: Load Data
print("\n[1/6] Loading data...")
df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
print(f"✓ Loaded {len(df)} employee records")

# STEP 2: Prepare Data
print("\n[2/6] Preparing data...")

df['Attrition'] = (df['Attrition'] == 'Yes').astype(int)
print(f"✓ Attrition cases: {df['Attrition'].sum()} ({df['Attrition'].mean()*100:.1f}%)")

# STEP 3: Feature Engineering (MATCHING DEPLOYMENT)
print("\n[3/6] Engineering features...")

df["IncomeToAgeRatio"] = df["MonthlyIncome"] / (df["Age"] + 1)
df["ExperienceToPromotionRatio"] = df["TotalWorkingYears"] / (df["YearsSinceLastPromotion"] + 1)
df["SatisfactionWorkloadRatio"] = df["JobSatisfaction"] / (df["WorkLifeBalance"] + 1)

df["PromotionGap"] = df["YearsSinceLastPromotion"] - df["YearsInCurrentRole"]
df["CareerStagnation"] = df["YearsInCurrentRole"] / (df["TotalWorkingYears"] + 1)

print("✓ Created 5 new features (aligned with deployment)")

# Encode categorical variables
categorical_cols = ['BusinessTravel', 'Department', 'EducationField', 
                   'Gender', 'JobRole', 'MaritalStatus', 'OverTime']

le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# Remove unnecessary columns
cols_to_drop = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
df = df.drop(columns=cols_to_drop, errors='ignore')

# Prepare X and y
X = df.drop('Attrition', axis=1)
y = df['Attrition']

feature_names = X.columns.tolist()
print(f"✓ Total features: {len(feature_names)}")

# STEP 4: Split Data
print("\n[4/6] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# STEP 5: Handle Imbalance
print("\n[5/6] Handling class imbalance with SMOTE...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

# STEP 6: Train Model
print("\n[6/6] Training XGBoost model...")
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='auc',
    use_label_encoder=False
)

model.fit(X_train_scaled, y_train_balanced, verbose=False)

# STEP 7: Evaluation
print("\n" + "=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print(f"\n🎯 Accuracy: {accuracy:.3f}")
print(f"🎯 ROC-AUC: {auc_score:.3f}")

# STEP 8: SHAP
print("\nCreating SHAP explainer...")
explainer = shap.TreeExplainer(model)

# STEP 9: Save Everything
print("\nSaving model files...")

joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(feature_names, 'feature_names.pkl')
joblib.dump(explainer, 'explainer.pkl')
joblib.dump(le_dict, 'label_encoders.pkl')

# ✅ Updated metrics (clean + UI ready)
metrics = {
    'accuracy': float(accuracy),
    'roc_auc': float(auc_score),
    'confusion_matrix': cm.tolist()
}

joblib.dump(metrics, 'metrics.pkl')

print("✅ All files saved successfully!")
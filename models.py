from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base


class Employee(Base):
    __tablename__ = "employees"

    id = Column(Integer, primary_key=True, index=True)
    age = Column(Integer)
    department = Column(String)
    job_role = Column(String)
    monthly_income = Column(Float)
    years_at_company = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey("employees.id"))
    risk_probability = Column(Float)
    risk_level = Column(String)
    model_version = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)


class ShapExplanation(Base):
    __tablename__ = "shap_explanations"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"))
    feature_name = Column(String)
    impact = Column(Float)
    direction = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)


class RecommendedAction(Base):
    __tablename__ = "recommended_actions"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"))
    action = Column(String)
    priority = Column(String)
    description = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)


class ModelMetadata(Base):
    __tablename__ = "model_metadata"

    id = Column(Integer, primary_key=True, index=True)
    model_version = Column(String)
    training_date = Column(DateTime)
    features_hash = Column(String)
    metrics_json = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)


class RiskTrend(Base):
    __tablename__ = "risk_trends"

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer)
    trend_type = Column(String)
    slope = Column(Float)
    window_size = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)


class DriftMetric(Base):
    __tablename__ = "drift_metrics"

    id = Column(Integer, primary_key=True, index=True)
    drift_type = Column(String, nullable=False)
    metric_name = Column(String, nullable=False)
    value = Column(Float, nullable=False)
    threshold = Column(Float, default=0.25)
    status = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
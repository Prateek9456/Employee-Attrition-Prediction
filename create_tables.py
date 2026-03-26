from database import engine
from models import (
    Employee,
    Prediction,
    ShapExplanation,
    RecommendedAction,
    ModelMetadata,
    RiskTrend,
    DriftMetric
)

Employee.__table__.create(bind=engine, checkfirst=True)
Prediction.__table__.create(bind=engine, checkfirst=True)
ShapExplanation.__table__.create(bind=engine, checkfirst=True)
RecommendedAction.__table__.create(bind=engine, checkfirst=True)
ModelMetadata.__table__.create(bind=engine, checkfirst=True)
RiskTrend.__table__.create(bind=engine, checkfirst=True)
DriftMetric.__table__.create(bind=engine, checkfirst=True)
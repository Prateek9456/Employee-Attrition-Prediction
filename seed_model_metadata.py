from sqlalchemy.orm import Session
from database import SessionLocal
from models import ModelMetadata
from datetime import datetime
import pickle


def seed_model():
    db: Session = SessionLocal()

    # Check if already exists
    existing = db.query(ModelMetadata).filter(
        ModelMetadata.model_version == "v1.0"
    ).first()

    if existing:
        print("Model v1.0 already exists.")
        return

    # Load metrics
    try:
        with open("metrics.pkl", "rb") as f:
            metrics = pickle.load(f)
    except:
        metrics = {"note": "metrics not found"}

    model_entry = ModelMetadata(
        model_version="v1.0",
        training_date=datetime.utcnow(),
        features_hash="v1_feature_contract",
        metrics_json=metrics
    )

    db.add(model_entry)
    db.commit()
    db.close()

    print("Model metadata seeded successfully.")


if __name__ == "__main__":
    seed_model()
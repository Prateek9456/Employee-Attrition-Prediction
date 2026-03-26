import os
import json
import joblib
from datetime import datetime

REGISTRY_PATH = "models"
METADATA_FILE = os.path.join(REGISTRY_PATH, "registry.json")


def _load_registry():
    if not os.path.exists(METADATA_FILE):
        return {"models": [], "active_model": None}

    with open(METADATA_FILE, "r") as f:
        return json.load(f)


def _save_registry(data):
    os.makedirs(REGISTRY_PATH, exist_ok=True)
    with open(METADATA_FILE, "w") as f:
        json.dump(data, f, indent=4)


def save_model(model, scaler, feature_names, encoders, metrics):
    os.makedirs(REGISTRY_PATH, exist_ok=True)

    version = f"v_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model_path = os.path.join(REGISTRY_PATH, version)
    os.makedirs(model_path, exist_ok=True)

    joblib.dump(model, os.path.join(model_path, "model.pkl"))
    joblib.dump(scaler, os.path.join(model_path, "scaler.pkl"))
    joblib.dump(feature_names, os.path.join(model_path, "feature_names.pkl"))
    joblib.dump(encoders, os.path.join(model_path, "label_encoders.pkl"))

    registry = _load_registry()

    registry["models"].append({
        "version": version,
        "path": model_path,
        "metrics": metrics,
        "created_at": datetime.now().isoformat()
    })

    _save_registry(registry)

    return version


def set_active_model(version):
    registry = _load_registry()
    registry["active_model"] = version
    _save_registry(registry)


def load_active_model():
    registry = _load_registry()
    active = registry.get("active_model")

    if not active:
        raise Exception("No active model set")

    model_path = os.path.join(REGISTRY_PATH, active)

    return {
        "model": joblib.load(os.path.join(model_path, "model.pkl")),
        "scaler": joblib.load(os.path.join(model_path, "scaler.pkl")),
        "feature_names": joblib.load(os.path.join(model_path, "feature_names.pkl")),
        "encoders": joblib.load(os.path.join(model_path, "label_encoders.pkl"))
    }


def get_active_metrics():
    registry = _load_registry()
    active = registry.get("active_model")

    for model in registry["models"]:
        if model["version"] == active:
            return model["metrics"]

    return None
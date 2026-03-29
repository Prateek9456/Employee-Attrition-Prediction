from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prediction_service import predict_and_explain
from llm_helper import generate_chat_response
from schemas import PredictionRequest, ChatRequest

print("🚀 APP STARTING...")

app = FastAPI()

# ✅ CORS (important for Streamlit)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "API is running"}


@app.post("/predict")
def predict(request: PredictionRequest):
    return predict_and_explain(request.dict())


@app.post("/chat")
def chat(request: ChatRequest):
    response = generate_chat_response(
        request.messages,
        request.employee_data,
        request.probability
    )

    return {
        "status": "success",
        "reply": response
    }
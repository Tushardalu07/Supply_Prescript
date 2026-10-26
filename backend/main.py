from fastapi import FastAPI

app = FastAPI(title="Supply Prescript API")

@app.get("/")
def home():
    return {"message": "Supply Prescript backend is running"}

@app.get("/health")
def health():
    return {"status": "ok"}
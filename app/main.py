from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import shutil
import tempfile
from pathlib import Path

from chicken_disease_classification.pipeline.prediction import PredictionPipeline

app = FastAPI(
    title="Chicken Disease Classification API",
    description="Upload a chicken fecal image to predict Coccidiosis or Healthy",
    version="1.0.0"
)

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Save to temp file (PredictionPipeline expects a filepath)
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=Path(file.filename).suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = Path(tmp.name)

    # Make prediction
    try:
        pipeline = PredictionPipeline(filename=tmp_path)
        result = pipeline.predict()
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    
    return JSONResponse(content={"prediction": result[0]["image"]})
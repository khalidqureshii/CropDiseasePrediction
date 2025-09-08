from fastapi import FastAPI, File, UploadFile # type: ignore
from fastapi.responses import JSONResponse # type: ignore
import uvicorn # type: ignore

from crop import analyze_disease  

app = FastAPI(title="Crop Disease Detection API")

@app.post("/analyze")
async def analyze_crop(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        result = analyze_disease(img_bytes)
        return JSONResponse(content={"result": result})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

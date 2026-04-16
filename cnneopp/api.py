from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import os

from .inference import run_cnneopp_pipeline

class PeptideRecord(BaseModel):
    hla: str
    peptide: str

class PredictRequest(BaseModel):
    records: List[PeptideRecord]
    
app = FastAPI(
    title="CNNeoPP Inference API",
    description="API wrapper for CNNeoPP consensus framework for neoantigen prioritization",
    version="0.1.0"
)

@app.get("/")
def health_check():
    return {"status": "ok", "message": "CNNeoPP API is running."}

@app.post("/predict")
def predict_neoantigen(request: PredictRequest):
    """
    接收 HLA 與 Peptide 資料陣列，回傳包含各個模型預測分數的結果。
    """
    if not request.records:
        raise HTTPException(status_code=400, detail="傳入的資料不可為空")

    try:
        # 將 Pydantic 模型轉為 DataFrame
        data_dicts = [record.model_dump() for record in request.records]
        input_df = pd.DataFrame(data_dicts)
        
        # 執行推論 (model_dir 預設為根目錄底下的 models)
        # 注意：請確保執行此 API 的工作目錄與 models 資料夾處於同一層，或以環境變數指定絕對路徑
        model_dir = os.environ.get("CNNEOPP_MODEL_DIR", "models")
        result_df = run_cnneopp_pipeline(input_df, model_dir=model_dir)
        
        # 轉成 JSON 格式回傳
        return {"results": result_df.to_dict(orient="records")}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

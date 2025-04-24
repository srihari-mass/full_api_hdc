from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import List
import requests
from paddleocr import PaddleOCR
from pdf2image import convert_from_path,convert_from_bytes
from PIL import Image
import numpy as np

app = FastAPI()
ocr = PaddleOCR(use_angle_cls=True, lang='en')
def extract_text_from_pdf(file: UploadFile):
    pdf_bytes = file.file.read()
    images = convert_from_bytes(pdf_bytes)
    full_text = ""

    for image in images:
        ocr_result = ocr.ocr(np.array(image))
        for line in ocr_result:
            for word_info in line:
                text = word_info[-1][0]
                full_text += text + " "
        full_text += "\n"
    return full_text.strip()
@app.post("/process")
async def process(
    file: UploadFile = File(...),
    llm_api_url: str = Form(...),
    desc_api_url: str = Form(...)
):
    try:
        text = extract_text_from_pdf(file)

        return JSONResponse(content={
            "text": text
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

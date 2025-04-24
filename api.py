from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import List
import requests
import networkx as nx
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from paddleocr import PaddleOCR
from pdf2image import convert_from_path
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



def extract_triples(text, llm_api_url):
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "messages": [
            {
                "role": "system",
                "content": """You are a clinical language model specialized in extracting medical diagnoses from unstructured text.

Your task is to extract **only the disease or diagnosis names** that are **explicitly stated** in the clinical note. Output each in the form of **TSV triples**:

Subject\tPredicate\tObject

Rules:
- Primarycondtion : always use "HasPrimaryCondition"
- Subject: always use "Patient"
- Predicate: always use "HasDiagnosis"
- Object: the disease or diagnosis as written in the text
- Only include **actual diseases, diagnoses, or medical conditions** mentioned in the note
- Do not include symptoms, lab values, procedures, or medications unless they are explicitly named as diagnoses
- No assumptions — extract only what is directly stated
- Output only the TSV lines with no extra text, explanations, or section headers
"""
            },
            {"role": "user", "content": text}
        ],
        "temperature": 0.0,
        "max_tokens": 12000,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0
    }
    response = requests.post(llm_api_url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content']


def create_graph(triples):
    G = nx.DiGraph()
    for line in triples.strip().split("\n"):
        parts = line.strip().split("\t")
        if len(parts) == 3:
            head, relation, tail = parts
            G.add_edge(head, tail, relation=relation)
    return G


def graph_to_base64(G):
    pos = nx.spring_layout(G)
    plt.figure(figsize=(10, 10))
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue",
            font_size=10, font_weight="bold", edge_color="gray", width=2)
    edge_labels = nx.get_edge_attributes(G, 'relation')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title("Medical Entity Relationships Graph")
    
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return img_base64


def parse_diseases(triples):
    diseases = []
    primary = []
    for line in triples.strip().split("\n"):
        parts = line.strip().split("\t")
        if len(parts) == 3:
            if parts[1] == "HasDiagnosis":
                diseases.append(parts[2])
            elif parts[1] == "HasPrimaryCondition":
                diseases.append(parts[2])
                primary.append(parts[2])
    return diseases, primary


def fetch_descriptions(diseases: List[str], desc_api_url: str):
    result = []
    for disease in diseases:
        try:
            response = requests.post(desc_api_url, json={"text": disease})
            if response.status_code == 200:
                try:
                    description = response.json()
                    result.append({"disease": disease, "description": description})
                except Exception:
                    result.append({"disease": disease, "description": "Invalid JSON returned"})
            else:
                result.append({"disease": disease, "description": f"Failed: {response.status_code}"})
        except Exception as e:
            result.append({"disease": disease, "description": f"Error: {str(e)}"})
    return result


@app.post("/process")
async def process(
    file: UploadFile = File(...),
    llm_api_url: str = Form(...),
    desc_api_url: str = Form(...)
):
    try:
        text = extract_text_from_pdf(file)
        triples = extract_triples(text, llm_api_url)
        graph = create_graph(triples)
        graph_img = graph_to_base64(graph)
        diseases, primary = parse_diseases(triples)
        descriptions = fetch_descriptions(diseases, desc_api_url)

        return JSONResponse(content={
            "triples": triples,
            "diseases": diseases,
            "primary": primary,
            "descriptions": descriptions,
            "graph_image_base64": graph_img
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

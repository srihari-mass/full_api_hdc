from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import List
import requests
from paddleocr import PaddleOCR
from pdf2image import convert_from_bytes
from PIL import Image
import numpy as np
from pydantic import BaseModel
import text2term
import networkx as nx
import uvicorn

app = FastAPI()

# Initialize OCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Request model for clinical note processing
class ClinicalNoteRequest(BaseModel):
    llm_api_url: str
    desc_api_url: str

# Extract text from PDF
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

# Extract diagnoses and create graphs
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

def parse_diseases(triples):
    diseases = []
    primary = []
    triples = triples.split('</think>')[-1]
    for line in triples.strip().split("\n"):
        parts = line.strip().split("\t")
        if len(parts) == 3:
            if parts[1] == "HasDiagnosis":
                diseases.append(parts[2])
            elif parts[1] == "HasPrimaryCondition":
                diseases.append(parts[2])
                primary.append(parts[2])
    print("test1")
    return {'diseases': diseases, 'primary': primary}

def full_des(diseases, desc_api_url):
    full = {}
    for disease in diseases:
        try:
            response = requests.post(desc_api_url, json={"text": disease})
            response.raise_for_status()
            description = response.json()['predictions'][0]['description']
            # Check for NaN values and replace them
            if description is None or isinstance(description, float) and math.isnan(description):
                description = "NO"
            full[disease] = description
        except Exception:
            full[disease] = "NO"
    return full

def mando(diseases):
    dfl = text2term.map_terms(
        source_terms=diseases,
        target_ontology="http://purl.obolibrary.org/obo/mondo.owl",
        max_mappings=1
    )
    map_terms = dfl['Mapped Term Label'].tolist()
    source_terms = dfl['Source Term'].tolist()
    urls = dfl['Mapped Term IRI'].tolist()
    return map_terms, source_terms, urls

def create_graph(triples):
    G = nx.DiGraph()
    for line in triples.strip().split("\n"):
        parts = line.strip().split("\t")
        if len(parts) == 3:
            head, relation, tail = parts
            G.add_edge(head, tail, relation=relation)
    return G

def enrich_graph_with_descriptions(G, full_descriptions):
    for disease, description in full_descriptions.items():
        G.add_edge(disease, description)
    return G

def enrich_graph_with_mondo(G, map_terms, source_terms, urls):
    for src, mapped in zip(source_terms, map_terms):
        G.add_edge(src, mapped)
    for mapped, url in zip(map_terms, urls):
        G.add_edge(mapped, url)
    return G

def graph_to_json(G):
    data = nx.readwrite.json_graph.node_link_data(G)
    return data

import math

@app.post("/process")
async def process(
    file: UploadFile = File(...),
    llm_api_url: str = Form(...),
    desc_api_url: str = Form(...),
    return_graph: bool = Form(False)
):
    try:
        # Extract text from the uploaded PDF
        text = extract_text_from_pdf(file)
        print(f"Extracted Text: {text[:500]}...")  # Log part of the extracted text

        # Extract triples (diseases) from the clinical note text
        triples = extract_triples(text, llm_api_url)
        print(f"Extracted Triples: {triples[:500]}...")  # Log part of the extracted triples

        # Parse diseases from the triples
        parsed = parse_diseases(triples)
        diseases = parsed['diseases']
        primary_diseases = parsed['primary']

        if not diseases:
            return JSONResponse(content={"message": "No diseases found in the note."})

        # Create the graph from the triples
        G = create_graph(triples)
        print(f"Created Graph: {G.nodes()}")  # Log the created graph's nodes

        # Get the full descriptions for the diseases
        full_desc = full_des(diseases, desc_api_url)
        print(f"Full Descriptions: {full_desc}")  # Log the full descriptions

       
        # Enrich the graph with descriptions
        G = enrich_graph_with_descriptions(G, full_desc)

        # Get MONDO mappings for the diseases
        map_terms, source_terms, urls = mando(diseases)
        print(f"MONDO Mappings: {map_terms}")  # Log the MONDO mappings

        # Enrich the graph with MONDO terms
        G = enrich_graph_with_mondo(G, map_terms, source_terms, urls)

        # Return the appropriate response based on `return_graph`
        if return_graph:
            graph_json = graph_to_json(G)
            return JSONResponse(content={"graph": graph_json})
        else:
            return JSONResponse(content={
                "primary_diseases": primary_diseases,
                "all_diseases": diseases,
                "full_descriptions": full_desc,
                "mondo_mappings": [
                    {"source_term": s, "mapped_term": m, "mondo_url": u}
                    for s, m, u in zip(source_terms, map_terms, urls)
                ]
            })
    except Exception as e:
        # Log the full error traceback for debugging
        import traceback
        print("Error occurred:", traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=2025)

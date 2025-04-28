from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import requests
import networkx as nx
import text2term
import uvicorn

app = FastAPI()

# Request model to accept clinical note text and URLs for APIs
class ClinicalNoteRequest(BaseModel):
    text: str
    llm_api_url: str
    desc_api_url: str

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
        "max_tokens": 22000,
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
    return {'diseases': diseases, 'primary': primary}

def create_graph(triples):
    G = nx.DiGraph()
    for line in triples.strip().split("\n"):
        parts = line.strip().split("\t")
        if len(parts) == 3:
            head, relation, tail = parts
            G.add_edge(head, tail, relation=relation)
    return G

def full_des(diseases, desc_api_url):
    full = {}
    for disease in diseases:
        try:
            response = requests.post(desc_api_url, json={"text": disease})
            response.raise_for_status()
            full[disease] = response.json()['predictions'][0]['description']
        except Exception:
            full[disease] = 'nan'
    return full

def enrich_graph_with_descriptions(G, full_descriptions):
    for disease, description in full_descriptions.items():
        G.add_edge(disease, description)
    return G

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

def enrich_graph_with_mondo(G, map_terms, source_terms, urls):
    for src, mapped in zip(source_terms, map_terms):
        G.add_edge(src, mapped)
    for mapped, url in zip(map_terms, urls):
        G.add_edge(mapped, url)
    return G

def graph_to_json(G):
    data = nx.readwrite.json_graph.node_link_data(G)
    return data

@app.post("/process_note/")
def process_clinical_note(request: ClinicalNoteRequest, return_graph: bool = Query(False)):
    try:
        # Extract triples from the clinical note using the provided LLM API URL
        triples = extract_triples(request.text, request.llm_api_url)
        
        # Parse diseases from the triples
        parsed = parse_diseases(triples)
        diseases = parsed['diseases']
        primary_diseases = parsed['primary']

        if not diseases:
            return {"message": "No diseases found in the note."}

        # Create the graph from the triples
        G = create_graph(triples)
        
        # Get the full descriptions for the diseases
        full_desc = full_des(diseases, request.desc_api_url)
        G = enrich_graph_with_descriptions(G, full_desc)

        # Get MONDO mappings for the diseases
        map_terms, source_terms, urls = mando(diseases)
        G = enrich_graph_with_mondo(G, map_terms, source_terms, urls)

        # Return graph or other details based on the `return_graph` flag
        if return_graph:
            graph_json = graph_to_json(G)
            return {"graph": graph_json}
        else:
            return {
                "primary_diseases": primary_diseases,
                "all_diseases": diseases,
                "full_descriptions": full_desc,
                "mondo_mappings": [
                    {"source_term": s, "mapped_term": m, "mondo_url": u}
                    for s, m, u in zip(source_terms, map_terms, urls)
                ]
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

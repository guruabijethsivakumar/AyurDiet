# 🌿 AyurDiet — Ayurvedic Personalised Diet Recommendation System

> **Smart India Hackathon (SIH) Project**  
> A hybrid **Graph RAG + Vector RAG** pipeline that generates personalised, Ayurveda-grounded diet plans using a Neo4j Knowledge Graph, FAISS Vector Store, and Google Gemini / Mistral LLMs.

---

## 📖 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Setup & Installation](#setup--installation)
- [Configuration](#configuration)
- [How to Run](#how-to-run)
- [Evaluation Pipeline](#evaluation-pipeline)
- [Data](#data)
- [Key Concepts](#key-concepts)
- [Results](#results)
- [Contributors](#contributors)

---

## Overview

**AyurDiet** bridges traditional Ayurvedic wisdom with modern AI to generate personalised 1-day diet plans. Given a user's:

- **Prakriti** (body constitution: Vata / Pitta / Kapha)
- **Age, Gender, Weight, Height** (BMI auto-computed)
- **Region** (South India, North India, Pan-India, etc.)
- **Season** (Summer, Winter, Monsoon, etc.)
- **Health conditions, activity level, sleep, stress**
- **Dietary preferences & nutrient deficiencies**

…the system retrieves contextually relevant foods from a **Neo4j Knowledge Graph** (Graph RAG) and **Ayurvedic rule embeddings** from a **FAISS Vector Store** (Vector RAG), then feeds them to an LLM to generate a structured, rationale-backed diet plan.

---

## Architecture

```
User Profile Input
       │
       ▼
┌──────────────────────────────────────────────────────┐
│                 HYBRID RAG PIPELINE                  │
│                                                      │
│  ┌──────────────────┐    ┌────────────────────────┐  │
│  │  Vector RAG      │    │  Graph RAG             │  │
│  │  (FAISS)         │    │  (Neo4j)               │  │
│  │                  │    │                        │  │
│  │  Ayurvedic Rules │    │  FoodNode graph with   │  │
│  │  (rasa, virya,   │    │  dosha, region, season │  │
│  │   vipaka, guna)  │    │  rasa/guna/virya/vipaka│  │
│  │                  │    │  nutrients, allergens  │  │
│  └────────┬─────────┘    └───────────┬────────────┘  │
│           │                         │               │
│           └────────────┬────────────┘               │
│                        ▼                            │
│              Combined Context (Rules + Foods)       │
└────────────────────────┬─────────────────────────────┘
                         │
                         ▼
              ┌──────────────────┐
              │  LLM Generation  │
              │  Gemini 2.5 Flash│
              │  or Mistral 7B   │
              └────────┬─────────┘
                       │
                       ▼
          Personalised 1-Day Diet Plan
   (Breakfast → Mid-Morning → Lunch → Snack → Dinner)
   with Ayurvedic rationale & nutrient breakdown
```

---

## Project Structure

```
sih/
├── graphRag_llm.py             # ✅ Main: Hybrid Graph+Vector RAG with Gemini
├── ayurveda_rag.py             # Vector RAG pipeline (FAISS + Excel + Gemini)
├── vector_rag.py               # Pure Vector RAG with Ollama (Mistral) + evaluation
├── vector_rag_bge.py           # Vector RAG using BGE embedding model variant
│
├── DeepEval.py                 # Graph RAG evaluation using Ollama (Mistral)
│
├── add_ayurveda.py             # Data enrichment: adds Ayurvedic attributes via Mistral
├── add_exception.py            # Adds virya/vipaka exception rules to dataset
├── virya_vipaka_exceptions.py  # Exception handling for Ayurvedic properties
│
├── generate_food_vector.py     # Builds FAISS food vector DB from dataset
├── generate_graph_vector.py    # Builds FAISS vector DB from ontology/graph data
├── vector_index_bge.py         # Builds FAISS index using BGE embeddings
│
├── foodnx_to_neo4j.py          # Imports food knowledge graph (GEXF) into Neo4j
├── ontologynx_to_neo4j.py      # Imports ontology graph (GEXF) into Neo4j
│
├── ollama_wrapper.py           # Utility wrapper for Ollama LLM calls
├── check.py                    # Quick sanity check script
├── modelcheck.py               # Model connectivity check
├── viualization.py             # Graph visualization utilities
│
├── indb.xlsx                   # Raw Indian food database (~1000+ items)
├── indb_filled_ayurveda.xlsx   # Enriched dataset with Ayurvedic attributes
├── foods_with_ayurveda_details.csv  # Processed food-Ayurveda mapping
│
├── food_kg.gexf                # Food Knowledge Graph (NetworkX GEXF)
├── ontology_kg.gexf            # Ayurveda Ontology Graph (NetworkX GEXF)
├── dosha_ontology.owl          # OWL Ontology for Dosha concepts
├── doshas.owl                  # Extended Dosha OWL ontology
├── check.rdf                   # RDF representation of ontology
├── updated_ontology.rdf        # Updated RDF ontology
│
├── ayurveda_index/             # FAISS index for Ayurvedic rules (MiniLM)
├── ayurveda_rules_index/       # FAISS index for ontology rules (Graph RAG)
├── food_vector_db/             # FAISS food vector DB (MiniLM)
├── food_vector_db_bge/         # FAISS food vector DB (BGE model)
├── combined_vector_db/         # Combined FAISS index
├── ontology_vector_db/         # FAISS index from ontology graph
│
├── deepeval_results.csv        # Graph RAG evaluation scores
├── vector_eval_results.csv     # Vector RAG evaluation scores (MiniLM)
├── vector_eval_results_bge.csv # Vector RAG evaluation scores (BGE)
├── graph_retrieval_metrics.csv # Graph retrieval performance metrics
│
├── knowledge-graph-llms/       # Reference submodule / research material
└── pellet/                     # Pellet OWL reasoner binaries
```

---

## Tech Stack

| Component | Technology |
|---|---|
| **Knowledge Graph** | Neo4j (Bolt) + NetworkX (GEXF) |
| **Vector Store** | FAISS (Facebook AI Similarity Search) |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` / `BAAI/bge-base-en` |
| **Primary LLM** | Google Gemini 2.5 Flash (`google-generativeai`) |
| **Alternative LLM** | Mistral 7B via Ollama (local) |
| **Ontology** | OWL / RDF (Pellet Reasoner) |
| **Data Processing** | Pandas, OpenPyXL |
| **RAG Framework** | LangChain (HuggingFace, FAISS, Text Splitters) |
| **Evaluation** | Custom LLM-as-judge metrics (Faithfulness, Precision, Recall, Relevancy) |
| **Language** | Python 3.10+ |

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- Neo4j Desktop or Neo4j Community Server running locally
- [Ollama](https://ollama.ai/) installed (for local Mistral inference)
- Google Gemini API key

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/sih.git
cd sih
```

### 2. Install Python Dependencies

```bash
pip install google-generativeai neo4j langchain langchain-huggingface \
            langchain-community faiss-cpu sentence-transformers \
            pandas openpyxl ollama networkx
```

### 3. Set Up Neo4j

1. Start your Neo4j instance (default: `bolt://localhost:7687`)
2. Import the food knowledge graph:
   ```bash
   python foodnx_to_neo4j.py
   ```
3. Import the ontology graph:
   ```bash
   python ontologynx_to_neo4j.py
   ```

### 4. Pull Mistral via Ollama (optional, for local evaluation)

```bash
ollama pull mistral:7b
```

---

## Configuration

### Environment Variables

```powershell
# PowerShell
$env:GOOGLE_API_KEY = "your_gemini_api_key_here"
```

```bash
# Bash / Linux / macOS
export GOOGLE_API_KEY="your_gemini_api_key_here"
```

### Neo4j Credentials

Edit `graphRag_llm.py` (lines 27–29):

```python
NEO4J_URI      = "bolt://localhost:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "your_neo4j_password"   # ← Change this
```

---

## How to Run

### Option A — Full Hybrid Graph+Vector RAG (Recommended)

Uses Neo4j for food retrieval + FAISS for Ayurvedic rules + Gemini for generation.

```bash
# Step 1: Build the Ayurvedic rules vector index (first time only)
python -c "from graphRag_llm import build_rule_index; build_rule_index()"

# Step 2: Run with example user profile
python graphRag_llm.py
```

**Customise the user profile** in `graphRag_llm.py` (bottom of file):

```python
user_inputs_example = {
    "age": 25,
    "gender": "Female",
    "weight": 58,
    "height": 162,
    "prakriti": "Vata",
    "health": "Anaemia",
    "activity": "moderate",       # sedentary / moderate / active
    "sleep": "irregular",
    "stress": "high",
    "region": "North India",
    "season": "Winter",
    "preferences": "Vegetarian",
    "nutrient_deficiency": "iron",
}
```

### Option B — Pure Vector RAG with Ollama

```bash
# Build food vector DB (first time)
python generate_food_vector.py

# Run Vector RAG evaluation across all dosha/season combos
python vector_rag.py
```

### Option C — Original Ayurveda RAG (Excel-based)

```bash
python ayurveda_rag.py
```

---

## Evaluation Pipeline

The system includes a rigorous **LLM-as-judge evaluation** framework with 4 metrics:

| Metric | Description |
|---|---|
| **Faithfulness** | Are all generated claims supported by the retrieved context? (No hallucinations) |
| **Contextual Precision** | Does the output use only the most relevant context? |
| **Contextual Recall** | Does the output cover all key relevant information? |
| **Answer Relevancy** | Does the output directly answer the user's query? |

### Run Graph RAG Evaluation

```bash
python DeepEval.py
# Output → deepeval_results.csv
```

### Run Vector RAG Evaluation

```bash
python vector_rag.py
# Output → vector_eval_results.csv

python vector_rag_bge.py
# Output → vector_eval_results_bge.csv
```

Results are saved as CSV files for comparison across embedding models and retrieval strategies.

---

## Data

### Indian Food Database (`indb.xlsx`)

- **1000+ Indian food items** with nutritional profiles
- Fields: `food_name`, `energy_kcal`, `carb_g`, `protein_g`, `fat_g`, `iron_mg`, `calcium_mg`, `sodium_mg`, `zinc_mg`, `region`, `season`

### Ayurvedic Enrichment (`indb_filled_ayurveda.xlsx`)

- Enriched dataset — each food annotated with:
  - `rasa` (taste: sweet, sour, pungent, bitter, astringent, salty)
  - `virya` (potency: heating/cooling)
  - `vipaka` (post-digestive effect)
  - `aggravates_dosha`, `pacifies_dosha`
  - `allergens`, `exceptions`

### Ontology Files

- `dosha_ontology.owl` / `doshas.owl` — OWL ontologies encoding Ayurvedic Dosha theory
- `food_kg.gexf` — Food Knowledge Graph (nodes = foods/doshas/seasons/regions, edges = relationships)
- `ontology_kg.gexf` — Ontology-based Knowledge Graph

---

## Key Concepts

| Ayurvedic Term | Meaning |
|---|---|
| **Prakriti** | Individual body constitution (Vata, Pitta, Kapha) |
| **Rasa** | Taste of food (Madhura/Sweet, Amla/Sour, Katu/Pungent, Tikta/Bitter, Kashaya/Astringent, Lavana/Salty) |
| **Virya** | Potency (Ushna/Heating or Sheeta/Cooling) |
| **Vipaka** | Post-digestive effect (Madhura, Amla, or Katu) |
| **Guna** | Quality (light, heavy, oily, dry, etc.) |
| **Dosha Balance** | "Like increases like; opposites balance" — core Ayurvedic dietary principle |

---

## Results

Evaluation results are stored in CSV files:

| File | Description |
|---|---|
| `deepeval_results.csv` | Graph RAG scores (3 doshas × 5 seasons = 15 combos) |
| `vector_eval_results.csv` | Vector RAG scores with MiniLM embeddings |
| `vector_eval_results_bge.csv` | Vector RAG scores with BGE embeddings |
| `graph_retrieval_metrics.csv` | Raw graph retrieval performance |

---

## License

This project was developed as part of the **Smart India Hackathon (SIH)**. All rights reserved.

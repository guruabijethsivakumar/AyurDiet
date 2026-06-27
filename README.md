# 🌿 AyurDiet — Ayurvedic Personalised Diet Recommendation System

> **Smart India Hackathon (SIH) Project**
> A hybrid **Graph RAG + Vector RAG** pipeline that generates personalised, Ayurveda-grounded diet plans using a Neo4j Knowledge Graph, FAISS Vector Store, and Google Gemini LLM — now with a **full clinical web portal** featuring Dietitian authentication, Google Sign-In, and patient profile management.

---

## 📖 Table of Contents

- [Overview](#overview)
- [What's New — Clinical Web Portal](#whats-new--clinical-web-portal)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Setup & Installation](#setup--installation)
- [Configuration](#configuration)
- [How to Run](#how-to-run)
- [Web Portal Usage](#web-portal-usage)
- [Evaluation Pipeline](#evaluation-pipeline)
- [Data](#data)
- [Key Concepts](#key-concepts)
- [Results](#results)
- [License](#license)

---

## Overview

**AyurDiet** bridges traditional Ayurvedic wisdom with modern AI to generate personalised 1-day diet plans. Given a patient's:

- **Patient Name** (new — personalised clinical output)
- **Prakriti** (body constitution: Vata / Pitta / Kapha)
- **Age, Gender, Weight, Height** (BMI auto-computed)
- **Region** (South India, North India, Pan-India, etc.)
- **Season** (Summer, Winter, Monsoon, etc.)
- **Health conditions, activity level, sleep, stress**
- **Dietary preferences & nutrient deficiencies**

…the system retrieves contextually relevant foods from a **Neo4j Knowledge Graph** (Graph RAG) and **Ayurvedic rule embeddings** from a **FAISS Vector Store** (Vector RAG), then feeds them to an LLM to generate a structured, rationale-backed, **patient-personalised** diet plan.

---

## ✨ What's New — Clinical Web Portal

The project now ships a **full-featured clinical web portal** built with Flask, HTML, CSS, and JavaScript:

### 🔐 Dietitian Authentication
- **Login Page** — Professional hospital-themed portal with username/password authentication.
- **Google Sign-In** — OAuth 2.0 integration via Google Identity Services. Dietitians can sign in using their Google account.
- **Session management** — Authentication state is stored securely in `localStorage` with a logout button always visible in the dashboard header.
- **Demo credentials** — `dietitian` / `password123` for quick access.

### 👤 Patient Profile Management
- **Patient Name** field added to the clinical form. The patient's name is passed through the full pipeline and embedded in the LLM prompt, resulting in personalised prescriptions (e.g. *"Ayurvedic Diet Prescription for Rahul Sharma"*).
- All other biometric, lifestyle, and dietary inputs remain as before.

### 🏥 Hospital-Themed UI
- Completely redesigned from a dark neon theme to a clean **white and clinical teal-green** hospital aesthetic.
- Premium typography, smooth animations, BMI calculator, and print-to-PDF support.

### 🌐 Web API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/` | Serves the clinical web portal |
| `GET`  | `/api/config` | Returns `GOOGLE_CLIENT_ID` for Sign-In |
| `POST` | `/api/login` | Standard username/password auth |
| `POST` | `/api/login-google` | Google OAuth token verification |
| `POST` | `/api/generate` | Generates the Ayurvedic diet plan |

---

## Architecture

```
Dietitian Login (Credentials / Google OAuth)
        │
        ▼
    Dashboard — Enter Patient Name + Profile
        │
        ▼
┌──────────────────────────────────────────────────────┐
│                 HYBRID RAG PIPELINE                  │
│                                                      │
│  ┌──────────────────┐    ┌────────────────────────┐  │
│  │  Vector RAG      │    │  Graph RAG             │  │
│  │  (FAISS)         │    │  (Neo4j Aura Cloud)    │  │
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
              └────────┬─────────┘
                       │
                       ▼
      Personalised 1-Day Diet Plan for [Patient Name]
   Section 1: Patient-facing meal table & guidelines
   Section 2: Dietitian clinical rationale & nutrients
```

---

## Project Structure

```
AyurDiet/
├── app.py                       # ✅ Flask web server — API endpoints & auth
├── graphRag_llm.py              # ✅ Main: Hybrid Graph+Vector RAG with Gemini
│
├── static/
│   ├── index.html               # ✅ Clinical web portal (login + dashboard)
│   ├── style.css                # ✅ Hospital-themed white & green UI
│   └── app.js                   # ✅ Auth logic, Google Sign-In, form handling
│
├── ayurveda_rag.py              # Vector RAG pipeline (FAISS + Excel + Gemini)
├── vector_rag.py                # Pure Vector RAG with Ollama (Mistral) + evaluation
├── vector_rag_bge.py            # Vector RAG using BGE embedding model variant
│
├── DeepEval.py                  # Graph RAG evaluation using Ollama (Mistral)
│
├── add_ayurveda.py              # Data enrichment: adds Ayurvedic attributes via Mistral
├── add_exception.py             # Adds virya/vipaka exception rules to dataset
├── virya_vipaka_exceptions.py   # Exception handling for Ayurvedic properties
│
├── generate_food_vector.py      # Builds FAISS food vector DB from dataset
├── generate_graph_vector.py     # Builds FAISS vector DB from ontology/graph data
│
├── foodnx_to_neo4j.py           # Imports food knowledge graph (GEXF) into Neo4j
├── ontologynx_to_neo4j.py       # Imports ontology graph (GEXF) into Neo4j
│
├── indb.xlsx                    # Raw Indian food database (~1000+ items)
├── indb_filled_ayurveda.xlsx    # Enriched dataset with Ayurvedic attributes
├── foods_with_ayurveda_details.csv  # Processed food-Ayurveda mapping
│
├── food_kg.gexf                 # Food Knowledge Graph (NetworkX GEXF)
├── ontology_kg.gexf             # Ayurveda Ontology Graph (NetworkX GEXF)
├── dosha_ontology.owl           # OWL Ontology for Dosha concepts
│
├── ayurveda_rules_index/        # FAISS index for Ayurvedic rules (MiniLM)
│
├── .env                         # Environment configuration (see below)
└── requirements.txt             # Python dependencies
```

---

## Tech Stack

| Component | Technology |
|---|---|
| **Web Framework** | Flask (Python) |
| **Frontend** | HTML5, Vanilla CSS, JavaScript (ES6+) |
| **Authentication** | Google Identity Services (OAuth 2.0) + custom credential auth |
| **Knowledge Graph** | Neo4j Aura Cloud (Bolt) + NetworkX (GEXF) |
| **Vector Store** | FAISS (Facebook AI Similarity Search) |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` |
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
- Neo4j Aura Cloud account (or local Neo4j Community Server)
- Google Gemini API key ([Get it at AI Studio](https://aistudio.google.com/))
- *(Optional)* Google Cloud project with OAuth 2.0 credentials for Google Sign-In
- *(Optional)* [Ollama](https://ollama.ai/) installed for local Mistral inference

### 1. Clone the Repository

```bash
git clone https://github.com/guruabijethsivakumar/AyurDiet.git
cd AyurDiet
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install flask google-generativeai neo4j langchain langchain-huggingface \
            langchain-community faiss-cpu sentence-transformers \
            pandas openpyxl python-dotenv networkx
```

### 3. Set Up Neo4j

1. Create a free [Neo4j Aura](https://neo4j.com/cloud/platform/aura-graph-database/) instance.
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

Create a `.env` file in the project root:

```env
# Google Gemini API Key (required)
GOOGLE_API_KEY=your_gemini_api_key_here

# Neo4j Aura Cloud credentials (required)
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password

# Google OAuth Client ID (optional — enables "Sign in with Google")
# Get from: https://console.cloud.google.com → APIs & Services → Credentials
GOOGLE_CLIENT_ID=your_client_id.apps.googleusercontent.com
```

### Setting Up Google Sign-In (Optional)

1. Go to [Google Cloud Console](https://console.cloud.google.com/) → **APIs & Services → Credentials**.
2. Click **Create Credentials → OAuth Client ID**.
3. Choose **Web application**.
4. Under **Authorised JavaScript origins**, add:
   - `http://127.0.0.1:5000`
   - `http://localhost:5000`
5. Copy the generated **Client ID** and paste it into your `.env` as `GOOGLE_CLIENT_ID`.

> If `GOOGLE_CLIENT_ID` is left blank, the app gracefully hides the Google Sign-In button and falls back to standard username/password login.

---

## How to Run

### Option A — Web Portal (Recommended)

Starts the full clinical web portal with login, patient profile form, and diet plan generation.

```bash
python app.py
```

Then open your browser at: **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

**Demo login credentials:**
- Username: `dietitian`
- Password: `password123`

### Option B — Full Hybrid RAG (Command Line)

```bash
# Step 1: Build the Ayurvedic rules vector index (first time only)
python -c "from graphRag_llm import build_rule_index; build_rule_index()"

# Step 2: Run with example user profile
python graphRag_llm.py
```

### Option C — Pure Vector RAG with Ollama

```bash
python generate_food_vector.py   # Build vector DB (first time)
python vector_rag.py             # Run evaluation
```

### Option D — Original Ayurveda RAG (Excel-based)

```bash
python ayurveda_rag.py
```

---

## Web Portal Usage

1. **Login** — Enter dietitian credentials or click **Sign in with Google**.
2. **Enter Patient Details** — Fill in the patient's name, Prakriti, biometrics, lifestyle, dietary preferences, and health conditions. Click **Demo Data** to auto-fill an example.
3. **Generate Diet Plan** — Click **Generate Diet Plan**. A progress overlay shows the pipeline stages (FAISS → Neo4j → Gemini).
4. **View Results** — The output contains two sections:
   - **Section 1 (Patient View):** Simple daily meal table addressed personally to the patient.
   - **Section 2 (Dietitian View):** Full clinical rationale with Ayurvedic properties, nutrient breakdown, and deficiency correction notes.
5. **Print / PDF** — Use the **Print / PDF** button to export the diet plan.
6. **Logout** — Click the logout icon in the header to return to the login screen.

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

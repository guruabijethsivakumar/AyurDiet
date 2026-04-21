import os
import subprocess
import json
import re
from typing import Dict, Any
from neo4j import GraphDatabase
import pandas as pd

# ==========================================================
# CONFIG
# ==========================================================

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "password"

DOSHAS = ["vata", "pitta", "kapha"]
SEASONS = ["Summer", "Winter", "Monsoon","Winter_and_Monsoon", "All_seasons"]
REGION = "Pan_India"

LLM_MODEL = "mistral:7b"


# ==========================================================
# FIXED OLLAMA CALL (UTF-8 safe on Windows)
# ==========================================================

def call_ollama(prompt: str) -> str:
    result = subprocess.run(
        ["ollama", "run", LLM_MODEL],
        input=prompt,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace"
    )
    return result.stdout.strip()


# ==========================================================
# CORRECT WORKING NEO4J QUERY
# ==========================================================

GRAPH_FOOD_QUERY = """
MATCH (f:FoodNode)-[:pacifiesDosha]->(d:FoodNode {label:$dosha})
OPTIONAL MATCH (f)-[:availableInRegion]->(reg)
OPTIONAL MATCH (f)-[:suitableForSeason]->(sea)
OPTIONAL MATCH (f)-[:hasRasa]->(r)
OPTIONAL MATCH (f)-[:hasGuna]->(g)
OPTIONAL MATCH (f)-[:hasVirya]->(v)
OPTIONAL MATCH (f)-[:hasVipaka]->(vp)
OPTIONAL MATCH (f)-[:hasAllergen]->(al)
OPTIONAL MATCH (f)-[:hasException]->(ex)

WHERE
    (
        $region IN ["Pan_India", "Pan_India_Urban", "Pan_India_with_regional_styles"]
        OR reg.label = $region
        OR reg.label IN ["Pan_India", "Pan_India_Urban", "Pan_India_with_regional_styles"]
    )
AND
    (
        $season = "All_seasons"
        OR sea.label = $season
    )

RETURN DISTINCT
    f.label AS food,
    collect(DISTINCT r.label) AS rasa,
    collect(DISTINCT g.label) AS guna,
    collect(DISTINCT v.label) AS virya,
    collect(DISTINCT vp.label) AS vipaka,
    collect(DISTINCT al.label) AS allergens,
    collect(DISTINCT ex.label) AS exceptions,
    f.Calories AS calories,
    f.carb_g AS carbs,
    f.Protein AS protein,
    f.Fat AS fat,
    f.iron_mg AS iron,
    f.calcium_mg AS calcium,
    f.sodium_mg AS sodium,
    f.zinc_mg AS zinc
ORDER BY food
"""


# ==========================================================
# RETRIEVE FROM NEO4J
# ==========================================================

def fetch_foods(dosha, season, region):
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    with driver.session() as session:
        result = session.run(GRAPH_FOOD_QUERY, {
            "dosha": dosha,
            "season": season,
            "region": region
        })
        return result.data()


# ==========================================================
# LLM GENERATOR
# ==========================================================

def llm_generate_diet(dosha, foods, season, region):
    food_list = ", ".join([f["food"] for f in foods])

    prompt = f"""
You are an Ayurvedic expert.

Dosha: {dosha}
Season: {season}
Region: {region}

Foods available: {food_list}

Generate a 4-meal Ayurvedic diet plan for {dosha}.
Explain the reasoning using rasa, guna, virya, vipaka.
"""

    return call_ollama(prompt)


# ==========================================================
# CONTEXT BUILDER
# ==========================================================

def build_context(foods):
    lines = []
    for f in foods:
        lines.append(
            f"""
FOOD: {f['food']}
Rasa: {f['rasa']}
Guna: {f['guna']}
Virya: {f['virya']}
Vipaka: {f['vipaka']}
Allergens: {f['allergens']}
Exceptions: {f['exceptions']}
"""
        )
    return "\n".join(lines)


# ==========================================================
# CUSTOM LOCAL METRICS USING OLLAMA (NO DEEPEVAL - FASTER, NO TIMEOUTS)
# ==========================================================

def evaluate_with_ollama(prompt: str, expected_format: str = "json") -> Dict[str, Any]:
    """
    Calls Ollama and parses response. Assumes JSON output for structured scoring.
    """
    response = call_ollama(prompt)
    if expected_format == "json":
        try:
            # Extract JSON from response (Mistral may add text; use regex to find JSON block)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"error": "No JSON found", "raw": response}
        except json.JSONDecodeError:
            return {"error": "Invalid JSON", "raw": response}
    return {"raw": response}


def compute_faithfulness(generated_output: str, ground_truth_context: str) -> float:
    """
    Faithfulness: Measures if generated output is faithful to context (no hallucinations).
    LLM scores 0-1 based on supported claims.
    """
    prompt = f"""
You are an evaluator. Assess faithfulness of the generated diet plan to the provided context.

Context: {ground_truth_context}

Generated Output: {generated_output}

Instructions:
- Identify key claims/facts in the output (e.g., food choices, rasa/guna explanations).
- Score 1.0 if ALL claims are directly supported by context without invention.
- Score 0.0 if many hallucinations.
- Output ONLY JSON: {{"faithfulness_score": <float 0-1>, "reason": "<brief explanation>"}}
"""
    result = evaluate_with_ollama(prompt)
    return result.get("faithfulness_score", 0.0)


def compute_contextual_precision(generated_output: str, ground_truth_context: str) -> float:
    """
    Contextual Precision: Measures if output uses only relevant parts of context (no irrelevant info).
    LLM scores 0-1 based on precision of used context.
    """
    prompt = f"""
You are an evaluator. Assess contextual precision of the generated diet plan.

Context: {ground_truth_context}

Generated Output: {generated_output}

Instructions:
- Check if output draws ONLY from relevant context elements (e.g., dosha-pacifying foods, attributes).
- Penalize inclusion of irrelevant or fabricated details.
- Score 1.0 for high precision (all used info is spot-on relevant).
- Score 0.0 for low precision (much noise/irrelevance).
- Output ONLY JSON: {{"precision_score": <float 0-1>, "reason": "<brief explanation>"}}
"""
    result = evaluate_with_ollama(prompt)
    return result.get("precision_score", 0.0)


def compute_contextual_recall(generated_output: str, ground_truth_context: str) -> float:
    """
    Contextual Recall: Measures if output covers key relevant info from context.
    LLM scores 0-1 based on recall of important elements.
    """
    prompt = f"""
You are an evaluator. Assess contextual recall of the generated diet plan.

Context: {ground_truth_context}

Generated Output: {generated_output}

Instructions:
- Identify key relevant items in context (e.g., top pacifying foods, their attributes for dosha/season).
- Score 1.0 if output comprehensively uses most key items.
- Score 0.0 if output misses most key info.
- Output ONLY JSON: {{"recall_score": <float 0-1>, "reason": "<brief explanation>"}}
"""
    result = evaluate_with_ollama(prompt)
    return result.get("recall_score", 0.0)


def compute_answer_relevancy(generated_output: str, query_text: str) -> float:
    """
    Answer Relevancy: Measures if output directly addresses the query.
    LLM scores 0-1 based on alignment to query intent.
    """
    prompt = f"""
You are an evaluator. Assess relevancy of the generated diet plan to the query.

Query: {query_text}

Generated Output: {generated_output}

Instructions:
- Check if output provides a direct, complete response (e.g., 4-meal plan with Ayurvedic reasoning).
- Score 1.0 if highly relevant and on-topic.
- Score 0.0 if off-topic or incomplete.
- Output ONLY JSON: {{"relevancy_score": <float 0-1>, "reason": "<brief explanation>"}}
"""
    result = evaluate_with_ollama(prompt)
    return result.get("relevancy_score", 0.0)


def compute_metrics(generated_output: str, ground_truth_context: str, query_text: str) -> Dict[str, float]:
    """
    Custom metrics using local Ollama (Mistral) - replacement for DeepEval.
    Returns dict with scores (0-1 scale for consistency). Faster than DeepEval (1-2 min per combo).
    """
    return {
        "faithfulness": compute_faithfulness(generated_output, ground_truth_context),
        "context_precision": compute_contextual_precision(generated_output, ground_truth_context),
        "context_recall": compute_contextual_recall(generated_output, ground_truth_context),
        "answer_relevancy": compute_answer_relevancy(generated_output, query_text),
    }


# ==========================================================
# MAIN LOOP
# ==========================================================

results = []

for dosha in DOSHAS:
    for season in SEASONS:

        print(f"\nEvaluating → DOSHA={dosha.upper()}  SEASON={season}")

        foods = fetch_foods(dosha, season, REGION)
        if not foods:
            print("No foods found — skipping.")
            continue

        context = build_context(foods)
        question = f"Generate a diet for {dosha} in {season}"

        generated = llm_generate_diet(dosha, foods, season, REGION)

        scores = compute_metrics(generated, context, question)

        row = {"dosha": dosha, "season": season, **scores}
        results.append(row)

        print("Scores:", row)


pd.DataFrame(results).to_csv("deepeval_results.csv", index=False)
print("\nSaved → deepeval_results.csv")
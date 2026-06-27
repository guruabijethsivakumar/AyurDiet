import os
from datetime import datetime
from typing import List, Dict

from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

import google.generativeai as genai
from neo4j import GraphDatabase
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter


# ======================================================
# CONFIG
# ======================================================

# --- Gemini ---
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise ValueError(
        "GOOGLE_API_KEY not set. Please set it in a .env file or in your environment."
    )
genai.configure(api_key=api_key)
llm_model = genai.GenerativeModel("models/gemini-2.5-flash")

# --- Neo4j ---
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# --- Vector DB ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RULES_INDEX_PATH = "ayurveda_rules_index"


# ======================================================
# AYURVEDIC RULE TEXT FOR VECTOR DB (NO EXCEL NEEDED)
# ======================================================

ONTOLOGY_RULE_CHUNKS = [
    # basic dosha + rasa/virya/vipaka rules
    "madhura (sweet) rasa: pacifies vata and pitta, but can aggravate kapha if heavy or excessive.",
    "amla (sour) rasa: pacifies vata, but can aggravate pitta and kapha.",
    "katu (pungent) rasa: reduces kapha, increases vata and pitta if overused.",
    "tikta (bitter) rasa: light, drying, pacifies pitta and kapha, can aggravate vata.",
    "kashaya (astringent) rasa: drying, cooling, pacifies pitta and kapha, may aggravate vata.",
    "lavana (salty) rasa: increases pitta and kapha, pacifies vata in moderation.",

    "ushna (heating) virya: stimulates digestion (agni), can aggravate pitta, often reduces kapha.",
    "sheeta (cooling) virya: pacifies pitta, can aggravate kapha and sometimes vata if too cold or damp.",

    "madhura vipaka (sweet post-digestive effect) nourishes tissues, can increase kapha.",
    "amla vipaka (sour post-digestive effect) tends to increase pitta.",
    "katu vipaka (pungent post-digestive effect) tends to reduce kapha but may aggravate vata.",

    "vata_type: energy of movement, light, cold, dry, subtle, mobile.",
    "pitta_type: energy of transformation, heat, metabolism, sharpness.",
    "kapha_type: energy of stability, structure, lubrication, heaviness.",

    "General rule: like increases like, opposites balance. "
    "For vata (cold, light, dry), use warm, moist, grounding foods. "
    "For pitta (hot, sharp), use cooling, mild, less oily foods. "
    "For kapha (heavy, slow), use light, dry, warming foods.",

    # Some example beverage rules (just conceptual, not exact foods)
    "Hot tea with heating spices typically has tikta and katu rasa, ushna virya, often aggravates pitta, "
    "can pacify kapha if used moderately.",
    "Cold sweet fruit juices tend to have madhura and amla rasa, sheeta virya, suitable for pitta but can aggravate kapha.",
    "Fermented, sour foods are usually heating in vipaka, and may increase pitta if overused.",
]


# ======================================================
# VECTOR RULE INDEX (FAISS)
# ======================================================

def build_rule_index(index_path: str = RULES_INDEX_PATH):
    """
    Build FAISS vector index from Ayurvedic rule text only.

    """
    print("Building Ayurvedic rules index...")

    # Split longer chunks if needed
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents(ONTOLOGY_RULE_CHUNKS)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(index_path)

    print(f"Rules index saved to {index_path}")


def load_rule_index(index_path: str = RULES_INDEX_PATH) -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    store = FAISS.load_local(index_path, embeddings=embeddings, allow_dangerous_deserialization=True)
    return store


# ======================================================
# NORMALIZATION HELPERS (MAPPING USER INPUT → GRAPH LABELS)
# ======================================================

def map_prakriti_to_dosha(prakriti: str) -> str:
    """
    Map user prakriti text to graph dosha label: 'vata', 'pitta', or 'kapha'.
    """
    if not prakriti:
        return "pitta"
    p = prakriti.strip().lower()

    # handle combos like "pitta-vata", "pitta vata", "pitta/kapha"
    for sep in ["-", "/", ","]:
        if sep in p:
            p = p.split(sep)[0].strip()
            break

    if p in ["vata", "pitta", "kapha"]:
        return p
    return p.split()[0]


def normalize_region(region: str) -> str:
    """
    Map human-readable region to graph label.
    Example Neo4j labels:
      East_India, West_India, North_India, South_India,
      Pan_India, Pan_India_Urban, Pan_India_with_regional_styles,
      Global_/_Western_influenced_Urban_India
    """
    if not region:
        return "Pan_India"

    r = region.strip()
    rl = r.lower()

    if rl in ["pan india", "pan-india"]:
        return "Pan_India"
    if rl in ["pan india urban", "pan-india urban"]:
        return "Pan_India_Urban"
    if "global" in rl:
        return "Global_/_Western_influenced_Urban_India"

    return r.replace(" ", "_")


def normalize_season(season: str) -> str:
    """
    Map human-readable season to graph label.
    Example Neo4j labels:
      All_seasons, Summer, Winter, Winter_and_Monsoon
    """
    if not season:
        return "All_seasons"

    s = season.strip()
    sl = s.lower()

    if "all" in sl:
        return "All_seasons"
    if "winter and monsoon" in sl or "winter_and_monsoon" in sl:
        return "Winter_and_Monsoon"

    return s.replace(" ", "_")


# ======================================================
# GRAPH RAG: NEO4J CYTHER QUERY
# ======================================================

GRAPH_FOOD_QUERY = """
MATCH (f:FoodNode)-[:pacifiesDosha]->(d:FoodNode {label:$dosha})
OPTIONAL MATCH (f)-[:availableInRegion]->(reg)
OPTIONAL MATCH (f)-[:suitableForSeason]->(sea)

WHERE 
    (reg IS NULL OR reg.label = $region)
AND
    (sea IS NULL OR sea.label = $season)

///////////////////////////////////////////////////////////////////////////
// RASA
///////////////////////////////////////////////////////////////////////////
CALL {
    WITH f
    OPTIONAL MATCH (f)-[:hasRasa]->(r)
    RETURN collect(DISTINCT r.label) AS rasa
}

///////////////////////////////////////////////////////////////////////////
// GUNA
///////////////////////////////////////////////////////////////////////////
CALL {
    WITH f
    OPTIONAL MATCH (f)-[:hasGuna]->(g)
    RETURN collect(DISTINCT g.label) AS guna
}

///////////////////////////////////////////////////////////////////////////
// VIRYA
///////////////////////////////////////////////////////////////////////////
CALL {
    WITH f
    OPTIONAL MATCH (f)-[:hasVirya]->(v)
    RETURN collect(DISTINCT v.label) AS virya
}

///////////////////////////////////////////////////////////////////////////
// VIPAKA
///////////////////////////////////////////////////////////////////////////
CALL {
    WITH f
    OPTIONAL MATCH (f)-[:hasVipaka]->(vp)
    RETURN collect(DISTINCT vp.label) AS vipaka
}

///////////////////////////////////////////////////////////////////////////
// ALLERGENS
///////////////////////////////////////////////////////////////////////////
CALL {
    WITH f
    OPTIONAL MATCH (f)-[:hasAllergen]->(al)
    RETURN collect(DISTINCT al.label) AS allergens
}

///////////////////////////////////////////////////////////////////////////
// EXCEPTIONS (not split, stored exactly as in Excel)
///////////////////////////////////////////////////////////////////////////
CALL {
    WITH f
    OPTIONAL MATCH (f)-[:hasException]->(ex)
    RETURN collect(ex.label) AS exceptions
}

///////////////////////////////////////////////////////////////////////////
// FINAL RETURN
///////////////////////////////////////////////////////////////////////////
RETURN DISTINCT
    f.label AS food,
    rasa,
    guna,
    virya,
    vipaka,
    allergens,
    exceptions,

    // NUTRIENTS
    f.Calories AS calories,
    f.carb_g AS carbs,
    f.Protein AS protein,
    f.Fat AS fat,
    f.iron_mg AS iron,
    f.calcium_mg AS calcium,
    f.sodium_mg AS sodium,
    f.zinc_mg AS zinc,
    f.unit_serving_energy_kcal AS serving_energy,
    f.unit_serving_carb_g AS serving_carbs,
    f.unit_serving_protein_g AS serving_protein,
    f.unit_serving_fat_g AS serving_fat,
    f.unit_serving_calcium_mg AS serving_calcium,
    f.unit_serving_sodium_mg AS serving_sodium,
    f.unit_serving_iron_mg AS serving_iron,
    f.unit_serving_zinc_mg AS serving_zinc

ORDER BY food;
"""


def query_graph_foods(user_inputs: Dict) -> List[Dict]:
    dosha = map_prakriti_to_dosha(user_inputs.get("prakriti", "pitta"))
    region = normalize_region(user_inputs.get("region", "Pan India"))
    season = normalize_season(user_inputs.get("season", "All seasons"))

    with driver.session() as session:
        records = session.run(
            GRAPH_FOOD_QUERY,
            dosha=dosha,
            region=region,
            season=season
        ).data()
    return records


def summarize_graph_food(record: Dict) -> str:
    """Convert a Neo4j record into a compact text line for the LLM."""

    def join_or_none(values):
        if not values:
            return "None"
        clean = [v for v in values if v]
        return ", ".join(clean) if clean else "None"

    return (
        f"Food: {record.get('food', '')} | "
        f"Rasa: {join_or_none(record.get('rasa'))} | "
        f"Guna: {join_or_none(record.get('guna'))} | "
        f"Virya: {join_or_none(record.get('virya'))} | "
        f"Vipaka: {join_or_none(record.get('vipaka'))} | "
        f"Allergens: {join_or_none(record.get('allergens'))} | "
        f"Exceptions: {join_or_none(record.get('exceptions'))} | "
        f"Per 100g approx -> "
        f"Calories: {record.get('calories')} kcal, "
        f"Carbs: {record.get('carbs')} g, "
        f"Protein: {record.get('protein')} g, "
        f"Fat: {record.get('fat')} g, "
        f"Iron: {record.get('iron')} mg, "
        f"Calcium: {record.get('calcium')} mg, "
        f"Sodium: {record.get('sodium')} mg, "
        f"Zinc: {record.get('zinc')} mg."
    )


# ======================================================
# HYBRID RETRIEVAL (RULES + GRAPH)
# ======================================================

def retrieve_context(user_inputs: Dict, rules_index_path: str = RULES_INDEX_PATH):
    """
    - VectorRAG: fetch Ayurvedic rules relevant to this user.
    - GraphRAG: fetch foods from Neo4j for this user's dosha/region/season.
    """
    # VectorRAG for rules
    rule_store = load_rule_index(rules_index_path)

    query = (
        f"Ayurvedic diet guidance for: Age {user_inputs['age']}, "
        f"Gender {user_inputs['gender']}, Prakriti {user_inputs['prakriti']}, "
        f"Season {user_inputs['season']}, Region {user_inputs['region']}, "
        f"Activity {user_inputs['activity']}, Preferences {user_inputs['preferences']}, "
        f"Nutrient deficiency {user_inputs['nutrient_deficiency']}."
    )

    rule_docs = rule_store.similarity_search(query, k=20)
    rules_texts = [doc.page_content for doc in rule_docs]

    # GraphRAG for foods
    graph_records = query_graph_foods(user_inputs)
    graph_food_summaries = [summarize_graph_food(rec) for rec in graph_records]
    print("\n================ GRAPH RAG RECORDS (NEO4J RESULTS) ================\n")
    for rec in graph_records:
        print(rec)
    print("\n===================================================================\n")


    return {
        "rules": rules_texts,
        "foods_structured": graph_food_summaries,
        "foods_raw": graph_records,
    }


# ======================================================
# DIET PLAN GENERATION
# ======================================================

def generate_diet_plan(user_inputs: Dict, rules_index_path: str = RULES_INDEX_PATH) -> str:
    ctx = retrieve_context(user_inputs, rules_index_path)

    weight = user_inputs["weight"]
    height_cm = user_inputs["height"]
    height_m = height_cm / 100.0
    bmi = weight / (height_m ** 2)

    if bmi < 18.5:
        bmi_cat = "underweight"
    elif bmi < 25:
        bmi_cat = "normal"
    elif bmi < 30:
        bmi_cat = "overweight"
    else:
        bmi_cat = "obese"

    activity_factor = {"sedentary": 2000, "moderate": 2500, "active": 3000}
    caloric_needs = activity_factor.get(user_inputs["activity"], 2200)
    if bmi_cat in ["overweight", "obese"]:
        caloric_needs = int(caloric_needs * 0.85)

    rules_text = "\n".join(ctx["rules"])
    foods_text = "\n".join(ctx["foods_structured"])

    now_str = datetime.now().strftime("%I:%M %p IST, %B %d, %Y")

    prompt = f"""
You are an Ayurvedic diet expert.

Use these Ayurvedic rules and concepts:
{rules_text}

Use ONLY these foods from the Neo4j knowledge graph
(already filtered by dosha, region, and season, with Ayurvedic and nutritional properties):
{foods_text}

User profile:
- Patient Name: {user_inputs.get('patient_name', 'Anonymous')}
- Age: {user_inputs['age']}
- Gender: {user_inputs['gender']}
- Weight: {user_inputs['weight']} kg
- Height: {user_inputs['height']} cm
- BMI: {bmi:.2f} ({bmi_cat})
- Prakriti (constitution): {user_inputs['prakriti']}
- Health conditions: {user_inputs['health']}
- Activity level: {user_inputs['activity']}
- Sleep pattern: {user_inputs['sleep']}
- Stress level: {user_inputs['stress']}
- Region: {user_inputs['region']}
- Season: {user_inputs['season']} (current date: {now_str})
- Preferences / restrictions: {user_inputs['preferences']}
- Key nutrient deficiency to address: {user_inputs['nutrient_deficiency']}

Instructions:
1. Use ONLY the foods listed above when constructing the plan.
2. Respect allergies and exceptions: if a food has 'Exceptions', use it carefully and explain in the rationale.
3. Use rasa, virya, and vipaka information to ensure the plan pacifies the dominant dosha ({user_inputs['prakriti']}).
4. Aim for approximately {caloric_needs} kcal/day, adjusted for BMI and activity.
5. Highlight foods that help correct the nutrient deficiency ({user_inputs['nutrient_deficiency']}) using the nutrients given.

Please structure the final output into two distinct, clearly labeled sections:

---

# SECTION 1: YOUR DIET PLAN (For the Patient)
*Design this for a common person to read and execute easily. Keep it clean, simple, and encouraging. Start this section with a personalized header addressing the patient by name (e.g. "Ayurvedic Diet Prescription for [Patient Name]").*

*   **Daily Meal Table / Timeline:** Create a simple table or timeline showing:
    *   **Meal (Time)**: Breakfast, Mid-Morning, Lunch, Evening Snack, Dinner.
    *   **What to Eat (Food Name)**: Clean up the database IDs into natural, friendly food names (e.g. use "Fruit Salad" instead of `Fruit_salad_Phalon_ka_salaad`, and "Oats & Chia Smoothie" instead of `Apple_oats_chia_seed_smoothie`).
    *   **Serving / Portion Size**: e.g., "1 Bowl", "1 Glass", "2 Cups". Use common terms.
*   **Simple Actionable Guidelines:** 3-4 bullet points in plain English telling them how to eat (e.g., "Eat in a quiet space", "Avoid cold drinks right after meals", "Keep hydrated"). Do NOT mention complex terms like Rasa, Virya, or Vipaka in this section.

---

# SECTION 2: DIETITIAN'S CLINICAL RATIONALE (For the Practitioner)
*Design this for professional dietitians/practitioners. Include all technical, chemical, and Ayurvedic details.*

*   **Total Daily Estimates:** Total Calories (kcal), Carbohydrates (g), Protein (g), Fat (g), and the specific targeted nutrient (e.g., Iron in mg).
*   **Meal-by-Meal Technical Breakdown:** For each of the 5 meals, specify:
    *   **Portion Size & Calories (kcal)**
    *   **Key Nutrient Values** (especially highlighting the targeted deficiency correction)
    *   **Ayurvedic Rationale:** Detailed explanation of the dosha balancing mechanism, Rasa (taste), Guna (quality), Virya (potency), and Vipaka (post-digestive effect) of the selected foods, and why they fit the current season ({user_inputs['season']}) and region ({user_inputs['region']}).
*   **Formulation Rationale:** A short summary of how the overall formula addresses the deficiency ({user_inputs['nutrient_deficiency']}) while maintaining Prakriti ({user_inputs['prakriti']}) equilibrium.
"""

    response = llm_model.generate_content(prompt)
    return response.text


# ======================================================
# MAIN (EXAMPLE RUN)
# ======================================================

if __name__ == "__main__":
    # 1. Build rule index once (if not already built)
    if not os.path.exists(RULES_INDEX_PATH):
        build_rule_index(RULES_INDEX_PATH)
    else:
        print(f"Using existing rules index at {RULES_INDEX_PATH}")

    # 2. Example user
    user_inputs_example = {
        "age": 25,
        "gender": "Male",
        "weight": 70,
        "height": 175,
        "prakriti": "Pitta",
        "health": "None",
        "activity": "moderate",   # sedentary / moderate / active
        "sleep": "regular",
        "stress": "low",
        "region": "South India",
        "season": "Summer",
        "preferences": "Vegetarian, no dairy",
        "nutrient_deficiency": "iron",
    }

    plan_text = generate_diet_plan(user_inputs_example, RULES_INDEX_PATH)
    print("\n================ DIET PLAN ================\n")
    print(plan_text)

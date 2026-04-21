import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
import google.generativeai as genai
import os

# Configure Gemini with fallback for API key
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not set. Please set it using $env:GOOGLE_API_KEY = 'your_key' in PowerShell.")
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.5-flash')

# Step 2: Parse and Index Data (Manual Rules)
def build_index(excel_file='indb.xlsx', index_path='ayurveda_index'):
    # Manually define ontology rules
    ontology_chunks = [
        "madhura (sweet, pacifies vata/pitta)",
        "amla (sour, pacifies vata but aggravates pitta/kapha)",
        "ushna (heating, aggravates pitta)",
        "sheeta (cooling, pacifies pitta)",
        "doshas: Dosha means Organization. As long as is normal, they maintain harmony.",
        "kapha_type: Energy of lubrication and structure.",
        "ojas: provides peace, calm and contentment",
        "pitta_type: Energy of transformation, digestion or metabolism",
        "prana: helps emotional harmony, balance and creativity",
        "tejas: courage, intellect, drive, radiance",
        "vata_type: Energy of Movement.",
        "Hot tea has rasa tikta and katu, virya ushna, vipaka katu, aggravates vata and pitta, pacifies kapha",
        "Instant coffee has rasa tikta and katu, virya ushna, vipaka katu, aggravates vata and pitta",
        "Iced tea has rasa tikta, virya sheeta, vipaka katu, aggravates vata",
        "Fruit punch has rasa madhura and amla, virya sheeta, vipaka madhura, pacifies vata, aggravates pitta",
        "Lemonade has rasa amla, virya sheeta, vipaka amla, pacifies vata, aggravates pitta and kapha",
        "Coco pine has rasa madhura, virya sheeta, vipaka madhura, pacifies pitta",
        "Banana milk has rasa madhura, virya sheeta, vipaka madhura, pacifies vata and pitta, aggravates kapha",
    ]
    
    # Parse Excel
    df = pd.read_excel(excel_file, sheet_name='Sheet1')
    food_docs = []
    for _, row in df.iterrows():
        food_str = f"Food: {row['food_name']}, Energy: {row['energy_kcal']} kcal, Carbs: {row['carb_g']} g, Protein: {row['protein_g']} g, Fat: {row['fat_g']} g"
        if 'region' in row:
            food_str += f", Region: {row['region']}"
        if 'season' in row:
            food_str += f", Season: {row['season']}"
        food_docs.append(food_str)
    
    documents = ontology_chunks + food_docs
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.create_documents([doc for doc in documents])
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(index_path)
    print(f"Index saved to {index_path}")

# Step 3: Retrieval
def retrieve_relevant_info(user_inputs, index_path='ayurveda_index'):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(index_path, embeddings=embeddings, allow_dangerous_deserialization=True)
    
    query = f"Generate diet plan for: Age {user_inputs['age']}, Gender {user_inputs['gender']}, Prakriti {user_inputs['prakriti']}, Season {user_inputs['season']}, Region {user_inputs['region']}, Activity {user_inputs['activity']}, Preferences {user_inputs['preferences']}. Use Ayurvedic principles to pacify {user_inputs['prakriti']} dosha."
    
    retrieved_docs = vectorstore.similarity_search(query, k=20)
    ontology_rules = [doc.page_content for doc in retrieved_docs if any(rule in doc.page_content.lower() for rule in ["pacifies", "aggravates", "madhura", "ushna", "sheeta"])]
    food_candidates = [doc.page_content for doc in retrieved_docs if "Food:" in doc.page_content]
    
    return {"rules": ontology_rules, "foods": food_candidates}

# Step 4: Generation
def generate_diet_plan(user_inputs, index_path='ayurveda_index'):
    retrieved = retrieve_relevant_info(user_inputs, index_path)
    
    bmi = user_inputs['weight'] / ((user_inputs['height'] / 100) ** 2)
    activity_factor = {'sedentary': 2000, 'moderate': 2500, 'active': 3000}
    caloric_needs = activity_factor.get(user_inputs['activity'], 2200)
    
    prompt = f"""
    You are an Ayurvedic diet expert. Use these rules: {retrieved['rules']}.
    Available foods (with nutritional data but no pre-assigned Ayurvedic properties): {retrieved['foods']}.
    
    User: Age {user_inputs['age']}, Gender {user_inputs['gender']}, BMI {bmi:.2f}, Prakriti {user_inputs['prakriti']}, Health {user_inputs['health']}, Activity {user_inputs['activity']}, Sleep {user_inputs['sleep']}, Stress {user_inputs['stress']}, Region {user_inputs['region']}, Season {user_inputs['season']} (current date: {pd.Timestamp.now().strftime('%I:%M %p IST, %B %d, %Y')}), Preferences {user_inputs['preferences']}.
    
    Infer the likely rasa, virya, and vipaka for each food based on its name and nutritional profile (e.g., high sugar = sweet rasa, cold drinks = cooling virya). Only use foods from the available foods list. Generate a 1-day diet plan (breakfast, lunch, snack, dinner) that pacifies the dominant dosha, matches season/region, respects preferences, and meets ~{caloric_needs} kcal. Include rationale based on inferred Ayurvedic properties and total calorie estimate.
    """
    
    response = model.generate_content(prompt)
    return response.text

# Step 5: Run the System
import os

if __name__ == "__main__":
    index_path = 'ayurveda_index'
    if not os.path.exists(index_path):
        build_index(index_path=index_path)
        print(f"Index built and saved at {index_path}")
    else:
        print(f"Loading existing index from {index_path}")

    user_inputs_list = [
        {
            'age': 25,
            'gender': 'Male',
            'weight': 70,
            'height': 175,
            'prakriti': 'Pitta',
            'health': 'None',
            'activity': 'moderate',
            'sleep': 'regular',
            'stress': 'low',
            'region': 'South India',
            'season': 'Summer',
            'preferences': 'Vegetarian, no dairy'
        },
       
    ]

    for user_inputs in user_inputs_list:
        plan = generate_diet_plan(user_inputs, index_path=index_path)
        print(f"\nGenerated Diet Plan for {user_inputs['prakriti']} at {pd.Timestamp.now().strftime('%I:%M %p IST, %B %d, %Y')}:\n")
        print(plan)
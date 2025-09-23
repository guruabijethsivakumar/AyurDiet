import os
import pandas as pd
import csv
from ollama import Client
import time
import re

# Initialize Ollama client and preload model
client = Client(host='http://localhost:11434')
client.generate(model='mistral:7b', prompt="Warm up", options={"num_threads": 12})  # Preload with max threads

# Function to get Ayurveda details for a single food item with retry mechanism
def get_ayurveda_details(food_name, max_retries=3):
    prompt = f"Be strict with Ayurveda terms format. Provide exactly: rasa, virya, vipaka, aggravates_dosha, pacifies_dosha, region, season for {food_name} as a single comma-separated list with no newlines, no descriptions, and exactly 7 values: rasa,virya,vipaka,aggravates_dosha,pacifies_dosha,region,season."

    for attempt in range(max_retries):
        response = client.generate(model='mistral:7b', prompt=prompt, options={"num_threads": 12})
        result_text = response['response'].strip()
        # Split on commas and clean up
        result = re.split(r',\s*', result_text.replace('\n', '').replace('\r', ''))
        result = [field.strip() for field in result if field.strip()]
        if len(result) == 7:
            print(f"Success for {food_name} (attempt {attempt+1}): {result}")
            return result
        else:
            print(f"Retry {attempt+1}/{max_retries} for {food_name} due to invalid output: {result_text}")

    # If all retries failed, fallback
    print(f"Failed to get valid response for {food_name} after {max_retries} attempts. Using defaults.")
    return ["", "", "", "", "", "", ""]

# File paths
input_xlsx = "C:/sih/indb.xlsx"
output_csv = "C:/sih/foods_with_ayurveda_details.csv"

# Read food names from XLSX
try:
    df = pd.read_excel(input_xlsx, engine="openpyxl", dtype=str)
    if "food_name" not in df.columns:
        raise ValueError("XLSX file must have a 'food_name' column.")
    food_names = df["food_name"].dropna().tolist()[1001:1015]  # Take from 1002 to 1015
    print(f"Loaded {len(food_names)} food items (1002–1015) from {input_xlsx}.")
except Exception as e:
    print(f"Error loading XLSX: {str(e)}. Using defaults.")
    food_names = ["rice", "chili", "milk"]

# Process each food item
start_time = time.time()
data = []
for i, food in enumerate(food_names, start=1002):
    ayurveda_values = get_ayurveda_details(food)
    row = [food] + ayurveda_values
    data.append(row)
    print(f"Processed record {i}/{1015}")

# Append to CSV (instead of overwriting)
file_exists = os.path.isfile(output_csv)
with open(output_csv, "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    if not file_exists:  # Only write header if file doesn't exist
        writer.writerow(["name", "rasa", "virya", "vipaka", "aggravates_dosha", "pacifies_dosha", "region", "season"])
    writer.writerows(data)

total_time = time.time() - start_time
print(f"Appended {len(food_names)} items to '{output_csv}'. Total time taken: {total_time:.2f} seconds.")

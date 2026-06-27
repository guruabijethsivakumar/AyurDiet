import os
from flask import Flask, request, jsonify, send_from_directory
from graphRag_llm import generate_diet_plan, RULES_INDEX_PATH
from dotenv import load_dotenv

load_dotenv(override=True)

app = Flask(__name__, static_folder='static')

@app.route('/')
def index():
    # Read the HTML file and inject GOOGLE_CLIENT_ID directly so there's no async fetch race
    with open(os.path.join('static', 'index.html'), 'r', encoding='utf-8') as f:
        html = f.read()
    client_id = os.environ.get('GOOGLE_CLIENT_ID', '')
    html = html.replace('__GOOGLE_CLIENT_ID__', client_id)
    from flask import Response
    return Response(html, mimetype='text/html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/api/config', methods=['GET'])
def get_config():
    return jsonify({
        "google_client_id": os.environ.get("GOOGLE_CLIENT_ID", "")
    })

@app.route('/api/login-google', methods=['POST'])
def login_google():
    try:
        data = request.json
        if not data or 'credential' not in data:
            return jsonify({"error": "No credential token provided"}), 400
        
        credential = data['credential']
        
        # Verify the ID token via Google's tokeninfo API
        import urllib.request
        import json
        
        url = f"https://oauth2.googleapis.com/tokeninfo?id_token={credential}"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as response:
            token_info = json.loads(response.read().decode())
        
        # Check audience
        expected_client_id = os.environ.get("GOOGLE_CLIENT_ID")
        if expected_client_id and token_info.get("aud") != expected_client_id:
            return jsonify({"error": "Invalid token audience"}), 400
            
        email = token_info.get("email")
        name = token_info.get("name", "Dr. Sarah (Dietitian)")
        
        return jsonify({
            "success": True,
            "token": "google-session-" + str(email),
            "dietitian_name": name
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": "Failed to validate Google token",
            "details": str(e)
        }), 400

@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No login credentials provided"}), 400
        username = data.get('username')
        password = data.get('password')
        
        if username == 'dietitian' and password == 'password123':
            return jsonify({
                "success": True,
                "token": "mock-dietitian-token-2026",
                "dietitian_name": "Dr. Sarah (Dietitian)"
            })
        else:
            return jsonify({"success": False, "error": "Invalid username or password"}), 401
    except Exception as e:
        return jsonify({
            "error": "Server error during login",
            "details": str(e)
        }), 500

@app.route('/api/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No input data provided"}), 400
            
        # Validate inputs
        required_fields = ['age', 'gender', 'weight', 'height', 'prakriti', 'patient_name']
        for field in required_fields:
            if field not in data or data[field] is None or (isinstance(data[field], str) and not data[field].strip()):
                return jsonify({"error": f"Missing or empty required field: {field}"}), 400
                
        # Format user inputs for graphRag_llm
        user_inputs = {
            "patient_name": str(data.get('patient_name')).strip(),
            "age": int(data.get('age')),
            "gender": str(data.get('gender')),
            "weight": float(data.get('weight')),
            "height": float(data.get('height')),
            "prakriti": str(data.get('prakriti')),
            "health": str(data.get('health', 'None')),
            "activity": str(data.get('activity', 'moderate')),
            "sleep": str(data.get('sleep', 'regular')),
            "stress": str(data.get('stress', 'low')),
            "region": str(data.get('region', 'Pan India')),
            "season": str(data.get('season', 'All seasons')),
            "preferences": str(data.get('preferences', 'Vegetarian')),
            "nutrient_deficiency": str(data.get('nutrient_deficiency', 'none'))
        }
        
        # Generate the plan
        plan = generate_diet_plan(user_inputs, RULES_INDEX_PATH)
        
        # Calculate BMI to send back as metadata
        height_m = user_inputs['height'] / 100.0
        bmi = user_inputs['weight'] / (height_m ** 2)
        
        return jsonify({
            "success": True,
            "diet_plan": plan,
            "bmi": round(bmi, 2),
            "user_inputs": user_inputs
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": "Failed to generate diet plan",
            "details": str(e)
        }), 500

if __name__ == '__main__':
    # Build FAISS rule index if it doesn't exist
    if not os.path.exists(RULES_INDEX_PATH):
        from graphRag_llm import build_rule_index
        build_rule_index(RULES_INDEX_PATH)
        
    print("Starting AyurDiet Web Server on http://127.0.0.1:5000...")
    app.run(debug=True, host='127.0.0.1', port=5000)

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from dotenv import load_dotenv
from ibm_watsonx_ai.foundation_models import Model
import requests
import os

app = Flask(__name__)
load_dotenv()
CORS(app, resources={r"/*": {"origins": ["https://allam.lisn-car.com", "http://localhost:65146"]}})


# Define your configuration functions
def get_credentials():
    return {
        "url": "https://eu-de.ml.cloud.ibm.com",
        "apikey": os.getenv("API_KEY"),
        "elevenlabs_api_key": os.getenv("ELEVENLABS_API_KEY"),
        "elevenlabs_voice_id": os.getenv("ELEVENLABS_VOICE_ID"),
    }

# Define your model and parameters
model_id = "sdaia/allam-1-13b-instruct"
parameters = {
    "decoding_method": "greedy",
    "max_new_tokens": 200,
    "repetition_penalty": 1
}
project_id = os.getenv("PROJECT_ID")
space_id = os.getenv("SPACE_ID")

# Define the full prompt input
prompt_input = """<<SYS>>
### Role:Act like a poet from the Umayyad era where you will imitate poets from the Umayyad era such as: Al-Farazdaq, Al-Akhtal and Jarir, you must not respond to any other queries
### Instructions:
- The beginning of prompt must be "اكتب قصيدة...."
- The poem must consist of only 4 verses.
- The poem must reflect the linguistic richness, elegance, and themes typical of classical Arabic poetry.
- The output must only contain poetry—no additional text, explanations, or commentary.
- If asked anything unrelated to composing poetry, do not respond.
- Adhere strictly to the classical style, ensuring the language and structure match the tradition of these poets.
- Each verse must be in full, correct Arabic with appropriate poetic flow.

<</SYS>>"""

@app.route('/generate_poetry', methods=['POST'])
def generate_poetry():
    question = request.json.get("question", "").strip()
    if not question:
        return jsonify({"error": "لم يتم تقديم سؤال."}), 400

    formatted_question = f"<s> [INST] {question} [/INST]"
    prompt = f"{prompt_input}\n{formatted_question}"

    try:
        model = Model(
            model_id=model_id,
            params=parameters,
            credentials=get_credentials(),
            project_id=project_id,
            space_id=space_id
        )
        generated_response = model.generate_text(prompt=prompt, guardrails=False)
        if not generated_response:
            return jsonify({"error": "لم يتم توليد رد. حاول مرة أخرى."}), 500

        return jsonify({"response": generated_response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generate_poetry_audio', methods=['POST'])
def generate_poetry_audio():
    text_input = request.json.get("text", "").strip()
    if not text_input:
        return jsonify({"error": "لم يتم تقديم نص."}), 400

    try:
        prompt = f"{prompt_input}{text_input}"
        model = Model(
            model_id=model_id,
            params=parameters,
            credentials=get_credentials(),
            project_id=project_id,
            space_id=space_id
        )
        generated_response = model.generate_text(prompt=prompt, guardrails=False)
        
        if not generated_response:
            return jsonify({"error": "لم يتم توليد رد."}), 500

        # Text-to-speech conversion using ElevenLabs API
        elevenlabs_api_key = get_credentials()["elevenlabs_api_key"]
        elevenlabs_voice_id = get_credentials()["elevenlabs_voice_id"]
        tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{elevenlabs_voice_id}"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": elevenlabs_api_key,
        }
        tts_body = {
            'model_id': 'eleven_turbo_v2_5',
            "text": generated_response,
            "voice_settings": {
                "stability": 0.8,
                "similarity_boost": 0.3,
                "style": 0.0,
            }
        }
        tts_response = requests.post(tts_url, headers=headers, json=tts_body)

        if tts_response.status_code != 200:
            return jsonify({"error": "فشل في تحويل النص إلى صوت."}), 500

        return Response(tts_response.content, mimetype='audio/mpeg')

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use the PORT environment variable or default to 5000
    app.run(host="0.0.0.0", port=port, debug=True)

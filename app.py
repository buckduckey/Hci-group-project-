import logging
import math
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from LLM import infer_from_data, generate_telemetry_from_persona, InferenceError

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)
CORS(app)

MAX_DAILY_HOURS = 24
MAX_PERSONA_CHARS = 1000


@app.route('/')
def index():
    return send_from_directory(BASE_DIR, 'index.html')


def parse_daily_hours(value):
    """Return Q1 as a number of hours in [0, 24], or None if it is not one."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        hours = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(hours) or not 0 <= hours <= MAX_DAILY_HOURS:
        return None
    return hours


def format_telemetry_prompt(telemetry):
    """Format stripped telemetry values into the 11-vector prompt for the blind LLM."""
    hours = telemetry['q1']
    return f"""
    Daily Digital Immersion: {hours:g} hours
    Late-night usage: {telemetry.get('q2')}
    Task Context: {telemetry.get('q3')}
    Interaction Cadence: {telemetry.get('q4')}
    Device Telemetry Tracked: {telemetry.get('q5')}
    Keystroke Biometrics Logged: {telemetry.get('q6')}
    Attention/Scroll Tracking: {telemetry.get('q7')}
    Facial Scans Active: {telemetry.get('q8')}
    Privacy Tooling: {telemetry.get('q9')}
    Dark Patterns Present: {telemetry.get('q10')}
    Cross-Device Geolocation Linked: {telemetry.get('q11')}
    """


@app.route('/api/infer', methods=['POST'])
def infer():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    hours = parse_daily_hours(data.get('q1'))
    if hours is None:
        return jsonify({"error": f"Q1 must be a number of hours between 0 and {MAX_DAILY_HOURS}."}), 400

    telemetry = dict(data, q1=hours)
    prompt_text = format_telemetry_prompt(telemetry)

    try:
        result = infer_from_data(prompt_text)
    except InferenceError as err:
        print(f"-> Inference failed: {err}")
        return jsonify({"error": str(err)}), 502

    return jsonify(result)


@app.route('/api/persona', methods=['POST'])
def simulate_persona():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    persona = data.get('persona')
    if not isinstance(persona, str) or not persona.strip():
        return jsonify({"error": "Please enter a short description of a person."}), 400
    if len(persona) > MAX_PERSONA_CHARS:
        return jsonify({"error": f"Persona description must be under {MAX_PERSONA_CHARS} characters."}), 400

    try:
        # Step 1: Translate persona description into raw telemetry numbers/booleans/enums
        telemetry = generate_telemetry_from_persona(persona.strip())

        # Step 2: Strip all text descriptors; pass ONLY the raw telemetry to the blind profiler
        stripped_prompt = format_telemetry_prompt(telemetry)
        inference = infer_from_data(stripped_prompt)
    except InferenceError as err:
        print(f"-> Persona simulation failed: {err}")
        return jsonify({"error": str(err)}), 502

    return jsonify({
        "telemetry": telemetry,
        "inference": inference,
    })


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    app.run(debug=True, port=5000)
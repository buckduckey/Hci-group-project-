import math

from flask import Flask, request, jsonify
from flask_cors import CORS
from LLM import infer_from_data, InferenceError

app = Flask(__name__)
CORS(app)

MAX_DAILY_HOURS = 24


def parse_daily_hours(value):
    """Return Q1 as a number of hours in [0, 24], or None if it is not one."""
    # bool is a subclass of int, so rule it out explicitly
    if isinstance(value, bool) or value is None:
        return None
    try:
        hours = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(hours) or not 0 <= hours <= MAX_DAILY_HOURS:
        return None
    return hours


@app.route('/api/infer', methods=['POST'])
def infer():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    hours = parse_daily_hours(data.get('q1'))
    if hours is None:
        return jsonify({"error": f"Q1 must be a number of hours between 0 and {MAX_DAILY_HOURS}."}), 400

    # Bundle the 11 questionnaire inputs into a text string for the LLM
    prompt_text = f"""
    Daily Digital Immersion: {hours:g} hours
    Late-night usage: {data.get('q2')}
    Task Context: {data.get('q3')}
    Interaction Cadence: {data.get('q4')}
    Device Telemetry Tracked: {data.get('q5')}
    Keystroke Biometrics Logged: {data.get('q6')}
    Attention/Scroll Tracking: {data.get('q7')}
    Facial Scans Active: {data.get('q8')}
    Privacy Tooling: {data.get('q9')}
    Dark Patterns Present: {data.get('q10')}
    Cross-Device Geolocation Linked: {data.get('q11')}
    """
    
    try:
        result = infer_from_data(prompt_text)
    except InferenceError as err:
        # Report the failure honestly instead of sending placeholder results
        print(f"-> Inference failed: {err}")
        return jsonify({"error": str(err)}), 502

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)

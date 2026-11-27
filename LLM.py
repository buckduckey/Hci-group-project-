import json
import math
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2:3b"
TIMEOUT_SECONDS = 90
MAX_OUTPUT_TOKENS = 600

ALLOWED_Q3 = ("Shopping", "Social", "News")
ALLOWED_Q4 = ("Deliberate", "Steady", "Rapid")
ALLOWED_Q9 = ("None", "AdBlock", "VPN")

SYSTEM_PROMPT = (
    "You are the profiling engine of a tracking system. From the 11 telemetry lines about one person, "
    "write 3 to 5 guesses about the PERSON: their sleep, mood, health, lifestyle, finances, or how easily "
    "they could be manipulated. Never copy an input line as a guess. "
    "info = the guess as a short sentence starting with 'Probably' or 'May'. "
    "evidence = cite the exact field names and their exact True, False, or numeric values from the input—"
    "never invert True to False or claim a True signal is absent. "
    "confidence = integer 0-100, lower for weaker guesses. "
    "conclusion = one sentence summarizing what kind of person this telemetry likely belongs to. "
    "Reply with JSON only."
)

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "key_info": {
            "type": "array",
            "minItems": 1,
            "maxItems": 6,
            "items": {
                "type": "object",
                "properties": {
                    "info": {"type": "string"},
                    "evidence": {"type": "string"},
                    "confidence": {"type": "integer"},
                },
                "required": ["info", "evidence", "confidence"],
            },
        },
        "conclusion": {"type": "string"},
    },
    "required": ["key_info", "conclusion"],
}

PERSONA_SYSTEM_PROMPT = (
    "You are a digital telemetry simulator. Given a short description of a person, "
    "translate their habits, schedule, tech savviness, and emotional state into realistic "
    "device tracking telemetry (q1 through q11). Follow these exact mapping rules:\n"
    "- q1: daily screen hours as a number from 0 to 24.\n"
    "- q2: MUST be true if the description mentions late night, midnight, 1 AM-5 AM, insomnia, or overnight; otherwise false.\n"
    "- q3: MUST be 'Shopping', 'Social', or 'News'.\n"
    "- q4: MUST be 'Rapid' (if impulsive, fast scrolling, switching tabs), 'Deliberate' (if slow, careful), or 'Steady'.\n"
    "- q5: background device hardware telemetry tracked (boolean; usually true unless privacy-hardened).\n"
    "- q6: keystroke/mouse biometrics logged (boolean).\n"
    "- q7: scroll velocity/hover attention tracked (boolean).\n"
    "- q8: webcam gaze/expression scans active (boolean).\n"
    "- q9: MUST be 'VPN' (if VPN/strong privacy tools mentioned), 'AdBlock' (if ad blocker mentioned), or 'None' (if default settings or no protection).\n"
    "- q10: MUST be true if ads, countdown timers, urgency, deals, or infinite scroll are mentioned; otherwise false.\n"
    "- q11: cross-device ID and IP geolocation linked (boolean).\n"
    "Reply with JSON only."
)

TELEMETRY_SCHEMA = {
    "type": "object",
    "properties": {
        "q1": {"type": "number"},
        "q2": {"type": "boolean"},
        "q3": {"type": "string", "enum": list(ALLOWED_Q3)},
        "q4": {"type": "string", "enum": list(ALLOWED_Q4)},
        "q5": {"type": "boolean"},
        "q6": {"type": "boolean"},
        "q7": {"type": "boolean"},
        "q8": {"type": "boolean"},
        "q9": {"type": "string", "enum": list(ALLOWED_Q9)},
        "q10": {"type": "boolean"},
        "q11": {"type": "boolean"},
    },
    "required": ["q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11"],
}

EXAMPLE_PERSONA_1 = "A privacy-conscious investigative journalist who reads articles carefully during the day using a VPN."
EXAMPLE_TELEMETRY_1 = json.dumps({
    "q1": 5.5,
    "q2": False,
    "q3": "News",
    "q4": "Deliberate",
    "q5": True,
    "q6": False,
    "q7": True,
    "q8": False,
    "q9": "VPN",
    "q10": False,
    "q11": False,
})

EXAMPLE_PERSONA_2 = "A stressed night-owl shopper scrolling deals late at night with default browser settings and countdown timers."
EXAMPLE_TELEMETRY_2 = json.dumps({
    "q1": 8.0,
    "q2": True,
    "q3": "Shopping",
    "q4": "Rapid",
    "q5": True,
    "q6": True,
    "q7": True,
    "q8": False,
    "q9": "None",
    "q10": True,
    "q11": True,
})

EXAMPLE_INPUT = """
    Daily Digital Immersion: 9 hours
    Late-night usage: True
    Task Context: Social
    Interaction Cadence: Rapid
    Device Telemetry Tracked: True
    Keystroke Biometrics Logged: False
    Attention/Scroll Tracking: True
    Facial Scans Active: False
    Privacy Tooling: None
    Dark Patterns Present: True
    Cross-Device Geolocation Linked: False
    """
EXAMPLE_OUTPUT = json.dumps({
    "key_info": [
        {
            "info": "Probably sleeps badly or keeps irregular hours.",
            "evidence": "Late-night usage is True and Daily Digital Immersion is 9 hours.",
            "confidence": 75,
        },
        {
            "info": "May be feeling lonely or seeking distraction.",
            "evidence": "Daily Digital Immersion is 9 hours with Task Context set to Social and Interaction Cadence set to Rapid.",
            "confidence": 55,
        },
        {
            "info": "Probably easy to nudge into impulsive actions.",
            "evidence": "Interaction Cadence is Rapid, Dark Patterns Present is True, and Privacy Tooling is None.",
            "confidence": 70,
        },
    ],
    "conclusion": "A heavy late-night social media user who may be tired, lonely, and vulnerable to impulsive nudges.",
})


class InferenceError(Exception):
    """The local model could not produce a usable answer."""


def _call_ollama(messages, schema, temperature=0):
    body = {
        "model": MODEL,
        "format": schema,
        "messages": messages,
        "stream": False,
        "keep_alive": "15m",
        "options": {"temperature": temperature, "num_predict": MAX_OUTPUT_TOKENS},
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            result = json.loads(response.read().decode())
    except Exception as err:
        raise InferenceError(
            f"Could not get an answer from Ollama ({err}). "
            f"Check that Ollama is running and the model '{MODEL}' is installed."
        ) from err

    if result.get("done_reason") == "length":
        raise InferenceError("The model's answer was cut off before it finished.")

    try:
        return json.loads(result["message"]["content"])
    except (KeyError, TypeError, ValueError) as err:
        raise InferenceError("The model did not return valid JSON.") from err


def generate_telemetry_from_persona(persona_text):
    """Step 1: Convert a user-written persona bio into raw, stripped telemetry."""
    messages = [
        {"role": "system", "content": PERSONA_SYSTEM_PROMPT},
        {"role": "user", "content": EXAMPLE_PERSONA_1},
        {"role": "assistant", "content": EXAMPLE_TELEMETRY_1},
        {"role": "user", "content": EXAMPLE_PERSONA_2},
        {"role": "assistant", "content": EXAMPLE_TELEMETRY_2},
        {"role": "user", "content": persona_text},
    ]
    raw = _call_ollama(messages, TELEMETRY_SCHEMA, temperature=0.2)
    return _validate_telemetry(raw)


def infer_from_data(data):
    """Step 2 (Blind): Infer personal traits strictly from stripped telemetry."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": EXAMPLE_INPUT},
        {"role": "assistant", "content": EXAMPLE_OUTPUT},
        {"role": "user", "content": data},
    ]
    output = _call_ollama(messages, RESPONSE_SCHEMA, temperature=0)
    return _validate(output)


def _validate_telemetry(raw):
    """Strip any unexpected fields and enforce strict primitive types."""
    if not isinstance(raw, dict):
        raise InferenceError("Generated telemetry was not in the expected format.")

    q1_raw = raw.get("q1")
    if isinstance(q1_raw, bool) or q1_raw is None:
        raise InferenceError("Generated screen hours were invalid.")
    try:
        q1 = round(float(q1_raw), 1)
    except (TypeError, ValueError) as err:
        raise InferenceError("Generated screen hours were invalid.") from err
    if not math.isfinite(q1):
        raise InferenceError("Generated screen hours were invalid.")
    q1 = max(0.0, min(24.0, q1))

    q3 = raw.get("q3")
    q4 = raw.get("q4")
    q9 = raw.get("q9")
    if q3 not in ALLOWED_Q3 or q4 not in ALLOWED_Q4 or q9 not in ALLOWED_Q9:
        raise InferenceError("Generated telemetry contained invalid categories.")

    clean = {"q1": q1, "q3": q3, "q4": q4, "q9": q9}
    for key in ("q2", "q5", "q6", "q7", "q8", "q10", "q11"):
        val = raw.get(key)
        if type(val) is not bool:
            raise InferenceError(f"Generated field {key} must be a boolean.")
        clean[key] = val

    return clean


def _validate(output):
    """Check the dossier answer has the shape the page expects and tidy the values."""
    if not isinstance(output, dict):
        raise InferenceError("The model's answer was not in the expected format.")

    items = output.get("key_info")
    conclusion = output.get("conclusion")
    if not isinstance(items, list) or not items or not isinstance(conclusion, str):
        raise InferenceError("The model's answer was not in the expected format.")

    key_info = []
    for item in items:
        if not isinstance(item, dict):
            raise InferenceError("The model's answer was not in the expected format.")
        try:
            confidence = int(item["confidence"])
            info = str(item["info"]).strip()
            evidence = str(item["evidence"]).strip()
        except (KeyError, TypeError, ValueError) as err:
            raise InferenceError("The model's answer was not in the expected format.") from err
        key_info.append({
            "info": info,
            "evidence": evidence,
            "confidence": max(0, min(100, confidence)),
        })

    return {"key_info": key_info, "conclusion": conclusion.strip()}
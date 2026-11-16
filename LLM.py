import json
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2:3b"
TIMEOUT_SECONDS = 60
# Upper bound on the reply length. Without it the model can get stuck repeating
# itself until it fills its whole context window, which takes minutes.
MAX_OUTPUT_TOKENS = 600

SYSTEM_PROMPT = (
    "You are the profiling engine of a tracking system. From the tracking data about one person, "
    "write 3 to 5 guesses about the PERSON: their sleep, mood, health, money, routines or how easily "
    "they could be manipulated. Never copy an input line as a guess. "
    "info = the guess as a short sentence starting with 'Probably' or 'May'. "
    "evidence = the input values that led to it. "
    "confidence = integer 0-100, lower for weaker guesses. "
    "conclusion = one sentence. Reply with JSON only."
)

# Ollama forces the reply to match this schema, so the structure the page
# expects is guaranteed rather than just asked for in the prompt.
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

# One worked example. The small model copies the input back as "inferences"
# unless it is shown what a real guess looks like. The input must use the same
# layout as the prompt built in app.py.
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
            "evidence": "Late-night usage is True and daily use is 9 hours.",
            "confidence": 70,
        },
        {
            "info": "May be feeling lonely or seeking distraction.",
            "evidence": "9 hours a day on social feeds with rapid, impulsive interaction.",
            "confidence": 45,
        },
        {
            "info": "Probably easy to nudge into impulsive actions.",
            "evidence": "Rapid interaction cadence while dark patterns are present and no privacy tooling is used.",
            "confidence": 60,
        },
    ],
    "conclusion": "A heavy late-night social media user who may be tired, lonely and easy to influence.",
})


class InferenceError(Exception):
    """The local model could not produce a usable answer."""


def infer_from_data(data):
    """Ask the local model what could be guessed from `data`.

    Returns {"key_info": [{"info", "evidence", "confidence"}, ...], "conclusion": str}.
    Raises InferenceError if the model can't be reached or its answer is unusable.
    There is deliberately no canned fallback: showing made-up results as if they
    came from the data would be misleading.
    """
    body = {
        "model": MODEL,
        "format": RESPONSE_SCHEMA,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": EXAMPLE_INPUT},
            {"role": "assistant", "content": EXAMPLE_OUTPUT},
            {"role": "user", "content": data},
        ],
        "stream": False,
        "options": {"temperature": 0, "num_predict": MAX_OUTPUT_TOKENS},
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
        output = json.loads(result["message"]["content"])
    except (KeyError, TypeError, ValueError) as err:
        raise InferenceError("The model did not return valid JSON.") from err

    return _validate(output)


def _validate(output):
    """Check the answer has the shape the page expects and tidy the values."""
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

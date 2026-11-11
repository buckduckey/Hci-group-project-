"""
Input string, returns llama3.2 processed information in dict.
Documentation: https://ollama.readthedocs.io/en/api/
"""

import json
import urllib.request
import urllib.error


def infer_from_data(data):
    requirement = """Utilize all given information in the data, list all personal or sensitive information that can be inferred from it. For each identified information, provide a confidence score that reflects the strength of evidence, with 0 being least confident and 100 being most confident. 
Reply as JSON: {"key_info": [{"info": "...", "evidence": "...", "confidence": 100}], "conclusion": "..."}"""

    body = {
        "model": "llama3.2",
        "format": "json",
        "messages": [
            {"role": "system", "content": requirement},
            {"role": "user", "content": data}
        ],
        "stream": False,
        "options": {"temperature": 0}
    }

    request = urllib.request.Request("http://localhost:11434/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.loads(response.read().decode())

            output = json.loads(result["message"]["content"])
            return output
    except urllib.error.URLError as err:
        print(err)
    except (json.JSONDecodeError, KeyError, TypeError) as err:
        print(err)

    return None
        

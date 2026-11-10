"""
Input string, returns llama3.2 processed information in dict.
Documentation: https://ollama.readthedocs.io/en/api/
"""

import json
import urllib.request


def model(data):
    requirement = """Utilize all given information in the data, list all personal or sensitive information that can be inferred from it.
Reply as JSON: {"key_info": [{"info": "...", "evidence": "..."}], "conclusion": "..."}"""

    body = {
        "model": "llama3.2",
        "format": "json",
        "messages": [
            {"role": "system", "content": requirement},
            {"role": "user", "content": data}
        ],
        "stream": False
    }

    request = urllib.request.Request("http://localhost:11434/api/chat", data = json.dumps(body).encode(), headers = {"Content-Type": "application/json"})
    response = urllib.request.urlopen(request)
    result = json.loads(response.read().decode())

    output = json.loads(result["message"]["content"])
    return output

# The Inference Mirror

HCI group project (Project 14). A prototype that shows what an AI could guess about a person from the data their devices collect, so people can see the consequences of sharing a data type before they agree to it.

The current prototype is a questionnaire about a simulated tracking setup. The answers are sent to a language model running locally through [Ollama](https://ollama.com), which returns a list of guesses with evidence and a confidence value.

No real personal data is used or stored. The model runs on your own machine and nothing is sent to an outside service.

## Requirements

- Python 3.10 or newer
- [Ollama](https://ollama.com/download)
- About 2 GB of free disk space for the model

## Setup

1. Install Ollama from <https://ollama.com/download>, then download the model:

   ```
   ollama pull llama3.2:3b
   ```

2. Install the Python packages (a virtual environment is recommended):

   ```
   python -m venv .venv
   .venv\Scripts\activate          # Windows
   source .venv/bin/activate       # macOS / Linux
   pip install -r requirements.txt
   ```

## Running

1. Make sure Ollama is running. The desktop app starts it automatically; otherwise run `ollama serve` in a separate terminal.

2. Start the app:

   ```
   python app.py
   ```

3. Open <http://localhost:5000> in a browser.

Open the page through that address, not by double-clicking `index.html`. The page calls the backend at the address it was loaded from, so it does not work when opened from disk.

Status messages (model requests, failures) are written to the terminal that runs `app.py`.

## Project layout

| File | Purpose |
| --- | --- |
| `app.py` | Flask server. Serves the page at `/` and the inference endpoint at `POST /api/infer`. |
| `LLM.py` | Sends the answers to the local model through Ollama and returns its guesses. |
| `index.html` | The questionnaire and the results view. |
| `requirements.txt` | Python dependencies. |

## Use of AI tools

- **In the product:** the inferences shown to the user are generated at run time by Meta's `llama3.2:3b` model, run locally through Ollama.
- **In development:** Claude Code (Anthropic) and Gemini (Google) were used to review the code against the brief and to write this README.

<!-- TODO (team): list any other AI tools used for code, design, data or the report, and what each was used for, before submission. -->

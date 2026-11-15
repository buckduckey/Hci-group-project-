"""Checks whether a model answer is grounded in the form input (issue #10).

`check_answer` compares each evidence line with the form values that were
actually sent and returns the mismatches it can detect mechanically:

- wrong_number   a number that is not the Q1 value
- example_leak   a wrong number that is the one used in the prompt's example
- wrong_value    a dropdown option that was not the one selected
- contradiction  a setting described as on when it was off, or the reverse
- no_citation    evidence that refers to nothing in the form

The wording of an answer is free text, so this is a heuristic: it catches the
kinds of mistake listed in the issue, but a clean result is not proof that an
answer is right. Read the answers as well.
"""
import re

import LLM

# Each yes/no question, with the phrases an evidence line uses to refer to it.
BOOLEAN_FIELDS = {
    "q2": ("Late-night usage", r"late[- ]night|at night|night[- ]?time"),
    "q5": ("Device Telemetry Tracked", r"telemetry|hardware profil"),
    "q6": ("Keystroke Biometrics Logged", r"keystroke|biometric|typing"),
    "q7": ("Attention/Scroll Tracking", r"scroll|attention tracking|attention/"),
    "q8": ("Facial Scans Active", r"facial|face scan|webcam"),
    "q10": ("Dark Patterns Present", r"dark pattern"),
    "q11": ("Cross-Device Geolocation Linked", r"geolocation|cross[- ]device|location"),
}
PRIVACY_TOOLING = r"privacy tool"

# Each dropdown question, with a pattern for every option it offers.
CHOICE_FIELDS = {
    "q3": ("Task Context", {
        "Shopping": r"shopping|e-?commerce",
        "Social": r"social",
        "News": r"\bnews\b",
    }),
    "q4": ("Interaction Cadence", {
        "Deliberate": r"deliberate",
        "Steady": r"steady",
        "Rapid": r"rapid",
    }),
    "q9": ("Privacy Tooling", {
        "AdBlock": r"ad[- ]?block",
        "VPN": r"\bvpn\b",
    }),
}

HOURS = r"hours?|immersion|daily"
NEGATION = (
    r"\b(no|not|none|never|without|false|lack\w*|absen\w*|disabled|inactive|off"
    r"|isn't|aren't|doesn't|don't|unlinked|untracked)\b"
)
AFFIRMATION = r"\btrue\b"
CLAUSE_BREAK = r"[,;.]|\band\b|\bwhile\b|\bbut\b|\bwith\b|\bdespite\b|\balthough\b"
NUMBER = r"\d+(?:\.\d+)?"


def example_hours():
    """The Q1 value in the prompt's worked example, read from LLM.py."""
    match = re.search(r"Immersion:\s*(" + NUMBER + ")", LLM.EXAMPLE_INPUT)
    return float(match.group(1)) if match else None


def _says_off(clause):
    """True if the clause describes a setting as off rather than on."""
    if re.search(AFFIRMATION, clause):
        return False
    return bool(re.search(NEGATION, clause))


def check_evidence(evidence, form):
    """Return the list of (kind, detail) problems found in one evidence line."""
    problems = []
    text = evidence.lower()
    cited = False

    actual_hours = float(form["q1"])
    for number in re.findall(NUMBER, text):
        cited = True
        if float(number) != actual_hours:
            kind = "example_leak" if float(number) == example_hours() else "wrong_number"
            problems.append((kind, f"says {number}, form says {form['q1']} hours"))
    if re.search(HOURS, text):
        cited = True

    for clause in re.split(CLAUSE_BREAK, text):
        off = _says_off(clause)

        for key, (label, pattern) in BOOLEAN_FIELDS.items():
            if re.search(pattern, clause):
                cited = True
                if off == bool(form[key]):
                    said = "off" if off else "on"
                    problems.append(("contradiction", f"{label} described as {said}, form says {form[key]}"))

        if re.search(PRIVACY_TOOLING, clause):
            cited = True
            has_tooling = form["q9"] != "None"
            if off == has_tooling:
                said = "absent" if off else "present"
                problems.append(("contradiction", f"Privacy Tooling described as {said}, form says {form['q9']}"))

        for key, (label, options) in CHOICE_FIELDS.items():
            for option, pattern in options.items():
                if re.search(pattern, clause):
                    cited = True
                    if option != form[key] and not off:
                        problems.append(("wrong_value", f"mentions {option}, but {label} is {form[key]}"))

    if not cited:
        problems.append(("no_citation", "refers to nothing in the form"))
    return problems


def check_answer(answer, form):
    """Return every problem in an answer as (item index, kind, detail, evidence)."""
    found = []
    for index, item in enumerate(answer["key_info"]):
        for kind, detail in check_evidence(item["evidence"], form):
            found.append((index, kind, detail, item["evidence"]))
    return found

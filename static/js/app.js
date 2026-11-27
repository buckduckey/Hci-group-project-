// questionnaire -> /api/infer -> report
// persona bio -> /api/persona -> stripped telemetry + blind report

const form = document.getElementById('profileForm');
const personaForm = document.getElementById('personaForm');
const personaInput = document.getElementById('personaInput');
const hoursInput = document.getElementById('q1');

const runButton = document.getElementById('runButton');
const buttonLabel = runButton.querySelector('.btn-label');
const runPersonaButton = document.getElementById('runPersonaButton');
const personaButtonLabel = runPersonaButton.querySelector('.btn-label');

const modeManualBtn = document.getElementById('modeManualBtn');
const modePersonaBtn = document.getElementById('modePersonaBtn');

const resultCard = document.getElementById('result');
const resultBody = document.getElementById('resultBody');
const reportMeta = document.getElementById('reportMeta');

const IDLE_LABEL = buttonLabel.textContent;
const PERSONA_IDLE_LABEL = personaButtonLabel.textContent;
const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

const TELEMETRY_LABELS = {
    q1: 'Screen hours/day',
    q2: 'Late-night (12–5 AM)',
    q3: 'Primary activity',
    q4: 'Browsing cadence',
    q5: 'Device telemetry',
    q6: 'Keystroke/mouse logs',
    q7: 'Scroll/hover tracking',
    q8: 'Webcam face scans',
    q9: 'Privacy tooling',
    q10: 'Dark patterns present',
    q11: 'Cross-device/IP linked'
};

modeManualBtn.addEventListener('click', () => switchMode('manual'));
modePersonaBtn.addEventListener('click', () => switchMode('persona'));

document.querySelectorAll('.preset-chip').forEach((chip) => {
    chip.addEventListener('click', () => {
        personaInput.value = chip.dataset.persona || '';
        personaInput.focus();
    });
});

form.addEventListener('submit', (event) => {
    event.preventDefault();
    runAI();
});

personaForm.addEventListener('submit', (event) => {
    event.preventDefault();
    runPersona();
});

function switchMode(mode) {
    const isPersona = mode === 'persona';
    form.hidden = isPersona;
    personaForm.hidden = !isPersona;
    modeManualBtn.setAttribute('aria-pressed', String(!isPersona));
    modePersonaBtn.setAttribute('aria-pressed', String(isPersona));
    resultCard.hidden = true;
}

function ticked(id) {
    return document.getElementById(id).checked;
}

function chosen(name) {
    return form.querySelector(`input[name="${name}"]:checked`).value;
}

async function runAI() {
    if (!hoursInput.reportValidity()) {
        return;
    }
    if (runButton.disabled || runPersonaButton.disabled) {
        return;
    }
    setBusy(true);

    const payload = {
        q1: Number(hoursInput.value),
        q2: ticked('q2'),
        q3: chosen('q3'),
        q4: chosen('q4'),
        q5: ticked('q5'),
        q6: ticked('q6'),
        q7: ticked('q7'),
        q8: ticked('q8'),
        q9: chosen('q9'),
        q10: ticked('q10'),
        q11: ticked('q11')
    };

    showWaiting('Reading your answers and drawing conclusions');

    try {
        const response = await fetch('/api/infer', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await response.json().catch(() => ({}));

        if (!response.ok) {
            showProblem(
                "The model couldn't put a profile together this time.",
                data.error || 'The server could not produce an inference.'
            );
            return;
        }

        showReport(data);
    } catch (err) {
        console.error(err);
        showProblem(
            "We can't reach the simulation on this computer.",
            'Check that app.py is running on port 5000 and that Ollama is open.'
        );
    } finally {
        setBusy(false);
    }
}

async function runPersona() {
    if (!personaInput.reportValidity()) {
        return;
    }
    if (runButton.disabled || runPersonaButton.disabled) {
        return;
    }
    const personaText = personaInput.value.trim();
    if (!personaText) {
        return;
    }

    setBusy(true);
    showWaiting('Step 1/2: Translating persona into raw telemetry, then running blind inference');

    try {
        const response = await fetch('/api/persona', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ persona: personaText })
        });

        const data = await response.json().catch(() => ({}));

        if (!response.ok) {
            showProblem(
                "The blind persona simulation couldn't finish.",
                data.error || 'The server could not complete both model passes.'
            );
            return;
        }

        showPersonaReport(personaText, data.telemetry, data.inference);
    } catch (err) {
        console.error(err);
        showProblem(
            "We can't reach the simulation on this computer.",
            'Check that app.py is running on port 5000 and that Ollama is open.'
        );
    } finally {
        setBusy(false);
    }
}

function setBusy(busy) {
    runButton.disabled = busy;
    runPersonaButton.disabled = busy;
    runButton.setAttribute('aria-busy', String(busy));
    runPersonaButton.setAttribute('aria-busy', String(busy));
    buttonLabel.textContent = busy ? 'Reading your answers…' : IDLE_LABEL;
    personaButtonLabel.textContent = busy ? 'Simulating & profiling…' : PERSONA_IDLE_LABEL;
}

function revealReport() {
    resultCard.hidden = false;
    resultCard.scrollIntoView({ behavior: reduceMotion.matches ? 'auto' : 'smooth', block: 'start' });
}

function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = String(text);
    return node;
}

function formatTimestamp() {
    const now = new Date();
    return `Prepared ${now.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' })}, ${now.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' })}`;
}

function showWaiting(message) {
    reportMeta.textContent = '';
    const line = el('p', 'typing', message || 'Reading your answers and drawing conclusions');
    line.append(el('span', 'caret'));
    line.lastChild.setAttribute('aria-hidden', 'true');

    resultBody.setAttribute('aria-busy', 'true');
    resultBody.replaceChildren(line);
    revealReport();
}

function buildDossierNodes(data) {
    const stamp = el('p', 'stamp', 'Guesses, not facts');
    const summary = el('p', 'summary', data.conclusion || 'The model finished without a summary.');

    let details;
    if (Array.isArray(data.key_info) && data.key_info.length > 0) {
        details = el('ol', 'inferences');
        details.setAttribute('aria-label', 'What the model guessed');
        details.replaceChildren(...data.key_info.map(makeInferenceItem));
    } else {
        details = el('p', 'empty', 'The model gave no specific guesses this time.');
    }

    return [stamp, summary, details];
}

function showReport(data) {
    reportMeta.textContent = formatTimestamp();
    resultBody.removeAttribute('aria-busy');
    resultBody.replaceChildren(...buildDossierNodes(data));
}

function showPersonaReport(personaText, telemetry, inference) {
    reportMeta.textContent = `${formatTimestamp()} · Blind simulation`;

    const stage1 = el('div', 'mirror-stage');
    stage1.append(
        el('p', 'mirror-kicker', '1. Hidden persona (withheld from profiler)'),
        el('p', '', personaText)
    );

    const stage2 = el('div', 'mirror-stage');
    stage2.append(
        el('p', 'mirror-kicker', '2. Stripped telemetry (all descriptors removed — only data sent to profiler)')
    );
    const grid = el('ul', 'telemetry-grid');
    Object.entries(TELEMETRY_LABELS).forEach(([key, label]) => {
        const li = el('li');
        const val = telemetry && key in telemetry ? String(telemetry[key]) : '—';
        li.append(el('span', '', label), el('strong', '', val));
        grid.append(li);
    });
    stage2.append(grid);

    const stage3 = el('div');
    stage3.append(
        el('p', 'mirror-kicker', '3. Blind AI reconstruction'),
        ...buildDossierNodes(inference || {})
    );

    resultBody.removeAttribute('aria-busy');
    resultBody.replaceChildren(stage1, stage2, stage3);
}

function showProblem(title, detail) {
    reportMeta.textContent = '';
    const box = el('div', 'note');
    box.setAttribute('role', 'alert');
    box.append(
        el('p', 'note-label', 'Something went wrong'),
        el('p', 'note-title', title),
        el('p', 'note-detail', detail),
        el('p', 'note-hint', 'Your inputs are still filled in, so you can try again.')
    );

    resultBody.removeAttribute('aria-busy');
    resultBody.replaceChildren(box);
    revealReport();
}

function describeConfidence(pct) {
    if (pct >= 70) return { level: 'high', word: 'very likely' };
    if (pct >= 40) return { level: 'med', word: 'plausible' };
    return { level: 'low', word: 'a long shot' };
}

function makeInferenceItem(item) {
    const raw = Number(item.confidence);
    const known = Number.isFinite(raw);
    const pct = known ? Math.round(Math.min(100, Math.max(0, raw))) : 0;
    const { level, word } = describeConfidence(pct);

    const li = el('li', 'inference');
    li.dataset.level = level;

    const head = el('div', 'inference-head');
    const leader = el('span', 'leader');
    leader.setAttribute('aria-hidden', 'true');
    head.append(
        el('span', 'inference-title', item.info ?? ''),
        leader,
        el('span', 'inference-pct', known ? `${pct}%` : '?')
    );

    const scale = el('div', 'inference-scale');
    const meter = el('div', 'meter');
    meter.setAttribute('role', 'meter');
    meter.setAttribute('aria-valuemin', '0');
    meter.setAttribute('aria-valuemax', '100');
    meter.setAttribute('aria-valuenow', String(pct));
    meter.setAttribute('aria-label', 'How sure the model is');
    const fill = el('div', 'meter-fill');
    fill.style.width = `${pct}%`;
    meter.append(fill);
    scale.append(meter, el('span', 'inference-word', known ? word : 'confidence not given'));

    const evidence = el('p', 'inference-evidence');
    evidence.append(el('span', '', 'based on: '), String(item.evidence ?? ''));

    li.append(head, scale, evidence);
    return li;
}
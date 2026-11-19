// questionnaire -> /api/infer -> report

const form = document.getElementById('profileForm');
const hoursInput = document.getElementById('q1');
const runButton = document.getElementById('runButton');
const buttonLabel = runButton.querySelector('.btn-label');
const resultCard = document.getElementById('result');
const resultBody = document.getElementById('resultBody');
const reportMeta = document.getElementById('reportMeta');

const IDLE_LABEL = buttonLabel.textContent;
const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

form.addEventListener('submit', (event) => {
    event.preventDefault();
    runAI();
});

function ticked(id) {
    return document.getElementById(id).checked;
}

function chosen(name) {
    return form.querySelector(`input[name="${name}"]:checked`).value;
}

async function runAI() {
    // Block blank, negative or impossible hours before anything is sent
    if (!hoursInput.reportValidity()) {
        return;
    }

    // One request at a time
    if (runButton.disabled) {
        return;
    }
    setBusy(true);

    // The keys and values here are the backend contract (see app.py and tests/)
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

    showWaiting();

    try {
        const response = await fetch('/api/infer', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await response.json().catch(() => ({}));

        // The backend reports model failures as an error status
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

function setBusy(busy) {
    runButton.disabled = busy;
    runButton.setAttribute('aria-busy', String(busy));
    buttonLabel.textContent = busy ? 'Reading your answers…' : IDLE_LABEL;
}

function revealReport() {
    resultCard.hidden = false;
    resultCard.scrollIntoView({ behavior: reduceMotion.matches ? 'auto' : 'smooth', block: 'start' });
}

function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    // Model output is untrusted text
    if (text !== undefined) node.textContent = String(text);
    return node;
}

function showWaiting() {
    reportMeta.textContent = '';
    const line = el('p', 'typing', 'Reading your answers and drawing conclusions');
    line.append(el('span', 'caret'));
    line.lastChild.setAttribute('aria-hidden', 'true');

    resultBody.setAttribute('aria-busy', 'true');
    resultBody.replaceChildren(line);
    revealReport();
}

function showReport(data) {
    const now = new Date();
    reportMeta.textContent = `Prepared ${now.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' })}, ${now.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' })}`;

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

    resultBody.removeAttribute('aria-busy');
    resultBody.replaceChildren(stamp, summary, details);
}

function showProblem(title, detail) {
    reportMeta.textContent = '';
    const box = el('div', 'note');
    box.setAttribute('role', 'alert');
    box.append(
        el('p', 'note-label', 'Something went wrong'),
        el('p', 'note-title', title),
        el('p', 'note-detail', detail),
        el('p', 'note-hint', 'Your answers are still filled in, so you can try again.')
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

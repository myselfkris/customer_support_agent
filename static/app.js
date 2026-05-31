/**
 * app.js
 * ------
 * Frontend logic for ReplyLens.
 *
 * This file does ONE thing: manages the WebSocket connection
 * between the browser and server.py.
 *
 * Flow:
 *   1. User fills form → clicks "Analyze"
 *   2. We open a WebSocket to ws://localhost:8000/ws/analyze
 *   3. We send { youtube_url, creator_email }
 *   4. Server sends progress messages → we update the UI
 *   5. Server sends "done" → we show the sheet link
 *   6. Server sends "error" → we show the error
 *
 * The page has 4 states:
 *   - form     (input fields visible)
 *   - progress (steps + progress bar visible)
 *   - result   (sheet link visible)
 *   - error    (error message visible)
 */

// ─── STATE ───────────────────────────────────────────────
let ws = null;

// ─── DOM REFERENCES ──────────────────────────────────────
const formSection     = document.getElementById('form-section');
const progressSection = document.getElementById('progress-section');
const resultSection   = document.getElementById('result-section');
const errorSection    = document.getElementById('error-section');

const urlInput        = document.getElementById('youtube-url');
const emailInput      = document.getElementById('creator-email');
const analyzeBtn      = document.getElementById('analyze-btn');

const stepsContainer  = document.getElementById('steps-container');
const progressBar     = document.getElementById('progress-bar');
const progressTitle   = document.getElementById('progress-title');
const progressSub     = document.getElementById('progress-sub');
const spinner         = document.getElementById('spinner');

const resultStats     = document.getElementById('result-stats');
const sheetLink       = document.getElementById('sheet-link');
const errorMessage    = document.getElementById('error-message');


// ─── UI STATE MANAGEMENT ─────────────────────────────────
function showSection(sectionId) {
    // Hide all sections
    [formSection, progressSection, resultSection, errorSection].forEach(s => {
        s.classList.add('hidden');
    });
    // Show the requested one
    document.getElementById(sectionId).classList.remove('hidden');
}


// ─── START ANALYSIS ──────────────────────────────────────
function startAnalysis() {
    const url   = urlInput.value.trim();
    const email = emailInput.value.trim();

    // Basic validation
    if (!url) {
        urlInput.focus();
        urlInput.style.borderColor = '#f87171';
        setTimeout(() => { urlInput.style.borderColor = ''; }, 2000);
        return;
    }
    if (!email || !email.includes('@')) {
        emailInput.focus();
        emailInput.style.borderColor = '#f87171';
        setTimeout(() => { emailInput.style.borderColor = ''; }, 2000);
        return;
    }

    // Switch to progress view
    showSection('progress-section');
    stepsContainer.innerHTML = '';
    progressBar.style.width = '0%';
    progressTitle.textContent = 'Starting...';
    progressSub.textContent = 'Connecting to server...';

    // Disable button to prevent double-click
    analyzeBtn.disabled = true;

    // Open WebSocket connection
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/analyze`;

    try {
        ws = new WebSocket(wsUrl);
    } catch (e) {
        showError('Could not connect to server. Is it running?');
        return;
    }

    ws.onopen = function() {
        // Connection established — send the YouTube URL and email
        ws.send(JSON.stringify({
            youtube_url: url,
            creator_email: email,
        }));
        addStep('extract', 'Connecting to YouTube...', 'loading');
    };

    ws.onmessage = function(event) {
        const msg = JSON.parse(event.data);
        handleProgress(msg);
    };

    ws.onerror = function() {
        showError('WebSocket connection failed. Make sure the server is running on port 8000.');
    };

    ws.onclose = function() {
        // Connection closed — nothing to do, UI already updated
    };
}


// ─── HANDLE PROGRESS MESSAGES ────────────────────────────
// Tracks which stages we've already added to the UI
const seenStages = new Set();

function handleProgress(msg) {
    const { stage, message, data } = msg;

    if (stage === 'error') {
        showError(message);
        return;
    }

    if (stage === 'done') {
        showResult(data);
        return;
    }

    // Update the progress title
    const stageLabels = {
        extract: '📥 Extracting Comments...',
        analyze: '🧠 Analyzing with AI...',
        upload:  '📊 Uploading to Sheets...',
    };
    progressTitle.textContent = stageLabels[stage] || 'Processing...';
    progressSub.textContent = message;

    // Update progress bar based on stage
    const stageProgress = {
        extract: 20,
        analyze: 50,
        upload:  85,
    };

    // If we have batch progress data, calculate more precise progress
    if (data && data.progress_pct && stage === 'analyze') {
        // Analyze goes from 20% to 80%
        const pct = 20 + (data.progress_pct * 0.6);
        progressBar.style.width = pct + '%';
    } else if (stageProgress[stage]) {
        progressBar.style.width = stageProgress[stage] + '%';
    }

    // Mark previous stages as done, add current stage
    if (!seenStages.has(stage)) {
        // Mark all previous steps as done
        document.querySelectorAll('.step.active').forEach(el => {
            el.classList.remove('active');
            el.classList.add('done');
            const icon = el.querySelector('.step-icon');
            if (icon) {
                icon.textContent = '✓';
                icon.classList.remove('loading');
                icon.classList.add('check');
            }
        });

        // Add new step
        addStep(stage, message, 'loading');
        seenStages.add(stage);
    } else {
        // Update existing step text
        updateLastStep(message);
    }
}


// ─── STEP UI HELPERS ─────────────────────────────────────
function addStep(stage, message, iconState) {
    const step = document.createElement('div');
    step.className = 'step active';
    step.dataset.stage = stage;

    const icon = document.createElement('span');
    icon.className = `step-icon ${iconState}`;
    icon.textContent = iconState === 'loading' ? '●' : '✓';

    const text = document.createElement('span');
    text.className = 'step-text';
    text.textContent = message;

    step.appendChild(icon);
    step.appendChild(text);
    stepsContainer.appendChild(step);

    // Scroll to bottom of steps
    stepsContainer.scrollTop = stepsContainer.scrollHeight;
}

function updateLastStep(message) {
    const steps = stepsContainer.querySelectorAll('.step');
    if (steps.length > 0) {
        const last = steps[steps.length - 1];
        const text = last.querySelector('.step-text');
        if (text) text.textContent = message;
    }
}


// ─── SHOW RESULT ─────────────────────────────────────────
function showResult(data) {
    progressBar.style.width = '100%';

    // Small delay for the progress bar to fill visually
    setTimeout(() => {
        showSection('result-section');

        if (data && data.sheet_url) {
            sheetLink.href = data.sheet_url;
        }

        // Show stats
        resultStats.innerHTML = '';
        if (data) {
            const stats = [
                { value: data.total_analyzed || '—', label: 'Comments' },
                { value: data.high_intent || '0',    label: 'High Intent' },
                { value: data.medium_intent || '0',  label: 'Medium Intent' },
            ];
            stats.forEach(s => {
                const el = document.createElement('div');
                el.className = 'stat';
                el.innerHTML = `
                    <div class="stat-value">${s.value}</div>
                    <div class="stat-label">${s.label}</div>
                `;
                resultStats.appendChild(el);
            });
        }
    }, 500);
}


// ─── SHOW ERROR ──────────────────────────────────────────
function showError(msg) {
    showSection('error-section');
    errorMessage.textContent = msg;
    analyzeBtn.disabled = false;
}


// ─── RESET ───────────────────────────────────────────────
function resetApp() {
    showSection('form-section');
    analyzeBtn.disabled = false;
    stepsContainer.innerHTML = '';
    progressBar.style.width = '0%';
    seenStages.clear();

    // Close WebSocket if still open
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.close();
    }
    ws = null;
}


// ─── KEYBOARD SHORTCUT ──────────────────────────────────
document.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !formSection.classList.contains('hidden')) {
        startAnalysis();
    }
});

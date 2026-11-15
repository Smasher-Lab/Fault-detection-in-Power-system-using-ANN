/**
 * Main Application Logic for Transmission Line Fault Detection & Classification
 * Department of Electrical and Electronics Engineering, NIT Puducherry
 * Team: Lalith (EE24B1030), Prabhat (EE24B1041), Vamsi (EE24B1040)
 */

// Global Application State
const state = {
    modelData: window.MODEL_DATA_FALLBACK || null,
    activeModelKey: 'baseline', // 'baseline', 'protective_standard', 'enhanced'
    currentInputs: {
        Ia: -650.4,
        Ib: -45.2,
        Ic: 18.5,
        Va: 0.14,
        Vb: -0.01,
        Vc: -0.15
    },
    currentActualFault: "LG (A-G)",
    currentActualDesc: "Single Line-to-Ground Fault on Phase A",
    currentSource: "Preset / IEEE Standard",
    sampleIndex: 0,
    isStreaming: false,
    streamInterval: null,
    streamSpeedMs: 1000,
    telemetryRecords: [],
    serverOnline: false,
    activeSlide: 1,
    totalSlides: 5
};

window.CURRENT_INPUTS = state.currentInputs;

// Visualizer references
let nnViz = null;
let scope = null;
let phasor = null;
let gridViz = null;

// Presets data
const PRESETS = {
    'healthy': {
        name: "Normal / Healthy (No Fault)",
        badge: "No Fault",
        vals: { Ia: -15.2, Ib: -12.4, Ic: 27.6, Va: 0.58, Vb: -0.28, Vc: -0.30 }
    },
    'lg': {
        name: "Single Line-to-Ground (LG: Phase A-G)",
        badge: "LG (A-G)",
        vals: { Ia: -650.4, Ib: -45.2, Ic: 18.5, Va: 0.14, Vb: -0.01, Vc: -0.15 }
    },
    'll': {
        name: "Line-to-Line (LL: Phase B-C)",
        badge: "LL (B-C)",
        vals: { Ia: 25.1, Ib: -580.3, Ic: 555.2, Va: 0.52, Vb: 0.08, Vc: -0.09 }
    },
    'llg': {
        name: "Double Line-to-Ground (LLG: Phase A-B-G)",
        badge: "LLG (A-B-G)",
        vals: { Ia: -590.2, Ib: -620.1, Ic: 42.3, Va: 0.12, Vb: 0.05, Vc: -0.25 }
    },
    'lll': {
        name: "3-Phase Symmetrical Short Circuit (LLL: A-B-C)",
        badge: "LLL (A-B-C)",
        vals: { Ia: 720.5, Ib: -810.2, Ic: 89.7, Va: 0.08, Vb: -0.04, Vc: 0.03 }
    },
    'lllg': {
        name: "3-Phase Symmetrical with Ground (LLLG: A-B-C-G)",
        badge: "LLLG (A-B-C-G)",
        vals: { Ia: 740.1, Ib: -805.4, Ic: 65.3, Va: 0.07, Vb: -0.05, Vc: 0.02 }
    }
};

const FEATURE_KEYS = ['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc'];

// Mathematical Forward Pass in Pure JavaScript (Zero-dependency Client-side Inference)
function runClientSideInference(rawInputs, modelKey) {
    if (!state.modelData) {
        state.modelData = window.MODEL_DATA_FALLBACK;
    }
    if (!state.modelData) return null;

    const modelObj = state.modelData.models[modelKey] || state.modelData.models['baseline'];
    const scaler = state.modelData.metadata.scaler;
    if (!modelObj || !scaler) return null;

    const x = [
        rawInputs.Ia,
        rawInputs.Ib,
        rawInputs.Ic,
        rawInputs.Va,
        rawInputs.Vb,
        rawInputs.Vc
    ];

    // 1. Z-Score Standardization
    const x_scaled = x.map((val, i) => (val - scaler.mean[i]) / scaler.scale[i]);

    // 2. Forward Propagation through 2-hidden layer MLP
    const arch = modelObj.architecture;
    const W1 = arch.weights[0];
    const b1 = arch.biases[0];
    const W2 = arch.weights[1];
    const b2 = arch.biases[1];
    const W3 = arch.weights[2];
    const b3 = arch.biases[2];

    // Layer 1: z1 = x * W1 + b1, a1 = ReLU(z1)
    const hidden1Size = b1.length;
    const z1 = new Array(hidden1Size).fill(0);
    const a1 = new Array(hidden1Size).fill(0);
    for (let j = 0; j < hidden1Size; j++) {
        let sum = b1[j];
        for (let i = 0; i < x_scaled.length; i++) {
            sum += x_scaled[i] * W1[i][j];
        }
        z1[j] = sum;
        a1[j] = Math.max(0, sum); // ReLU
    }

    // Layer 2: z2 = a1 * W2 + b2, a2 = ReLU(z2)
    const hidden2Size = b2.length;
    const z2 = new Array(hidden2Size).fill(0);
    const a2 = new Array(hidden2Size).fill(0);
    for (let k = 0; k < hidden2Size; k++) {
        let sum = b2[k];
        for (let j = 0; j < hidden1Size; j++) {
            sum += a1[j] * W2[j][k];
        }
        z2[k] = sum;
        a2[k] = Math.max(0, sum); // ReLU
    }

    // Layer 3: z3 = a2 * W3 + b3, probs = Softmax(z3)
    const outSize = b3.length;
    const z3 = new Array(outSize).fill(0);
    let maxZ = -Infinity;
    for (let m = 0; m < outSize; m++) {
        let sum = b3[m];
        for (let k = 0; k < hidden2Size; k++) {
            sum += a2[k] * W3[k][m];
        }
        z3[m] = sum;
        if (sum > maxZ) maxZ = sum;
    }

    // Softmax
    const expZ = z3.map(v => Math.exp(v - maxZ));
    const sumExp = expZ.reduce((a, b) => a + b, 0);
    const probs = expZ.map(v => v / sumExp);

    let bestIdx = 0;
    let bestProb = probs[0];
    for (let i = 1; i < probs.length; i++) {
        if (probs[i] > bestProb) {
            bestProb = probs[i];
            bestIdx = i;
        }
    }

    const predictedClass = modelObj.classes[bestIdx];

    // Power Systems Physics Analysis
    const ia = rawInputs.Ia;
    const ib = rawInputs.Ib;
    const ic = rawInputs.Ic;
    const va = rawInputs.Va;
    const vb = rawInputs.Vb;
    const vc = rawInputs.Vc;
    const ig = ia + ib + ic;
    const vg = va + vb + vc;

    const explanations = [];
    if (predictedClass.includes("No Fault")) {
        explanations.push("Balanced 3-Phase currents and voltages operating normally within rated capacity.");
        explanations.push(`Ground residual leakage current Ig = ${ig.toFixed(2)} A (negligible).`);
    } else if (predictedClass.includes("LG")) {
        explanations.push(`High ground fault return current detected: Ig = ${ig.toFixed(2)} A.`);
        explanations.push(`Phase A current surges to ${ia.toFixed(1)} A while phase voltage collapses to ${va.toFixed(3)} p.u.`);
        explanations.push(`Classic Single Line-to-Ground (A-G) signature.`);
    } else if (predictedClass.includes("LLG")) {
        explanations.push(`Simultaneous Phase A (${ia.toFixed(1)} A) & Phase B (${ib.toFixed(1)} A) current surges.`);
        explanations.push(`Significant zero-sequence ground fault current: Ig = ${ig.toFixed(2)} A.`);
        explanations.push(`Double Line-to-Ground (A-B-G) short circuit.`);
    } else if (predictedClass.includes("LL")) {
        explanations.push(`Equal and opposite phase surge currents on Phase B (${ib.toFixed(1)} A) and Phase C (${ic.toFixed(1)} A).`);
        explanations.push(`Ground current Ig = ${ig.toFixed(2)} A is near zero (isolated phase-to-phase contact).`);
        explanations.push(`Phase A remains energized and healthy.`);
    } else if (predictedClass.includes("LLL")) {
        explanations.push(`Severe symmetrical 3-Phase short circuit across all conductors (A-B-C).`);
        explanations.push(`Massive current surge on all 3 phases with phase voltages depressed near zero.`);
        explanations.push(`Kirchhoff's Current Law holds: sum Ia + Ib + Ic = ${ig.toFixed(2)} A ≈ 0 A due to 120° physical symmetry.`);
    }

    return {
        predicted_fault: predictedClass,
        confidence: (bestProb * 100).toFixed(2),
        probabilities: probs,
        activations: {
            inputs_scaled: x_scaled.map(v => +v.toFixed(4)),
            layer1_z: z1.map(v => +v.toFixed(4)),
            layer1_a: a1.map(v => +v.toFixed(4)),
            layer2_z: z2.map(v => +v.toFixed(4)),
            layer2_a: a2.map(v => +v.toFixed(4)),
            layer3_z: z3.map(v => +v.toFixed(4)),
            layer3_probs: probs.map(v => +v.toFixed(4))
        },
        power_physics: {
            ground_current_ig: +ig.toFixed(2),
            zero_sequence_voltage_vg: +vg.toFixed(4),
            electrical_explanation: explanations
        }
    };
}

// Electrical Engineering Physical Fault Classifier (Ground Truth Inferrer)
function inferPhysicalFault(inputs) {
    const { Ia, Ib, Ic, Va, Vb, Vc } = inputs;
    const maxI = Math.max(Math.abs(Ia || 0), Math.abs(Ib || 0), Math.abs(Ic || 0));
    if (maxI < 120) {
        return {
            type: "No Fault (Healthy)",
            desc: "Continuous load within safe line thermal capacity",
            guidance: "Normal balanced 3-phase power transmission."
        };
    }

    const ig = Math.abs((Ia || 0) + (Ib || 0) + (Ic || 0));
    const isGroundInvolved = ig > 60;

    if (!isGroundInvolved) {
        // Line-to-Line (No ground return)
        if (Math.abs(Ia) > 200 && Math.abs(Ib) > 200 && Math.abs(Ic) < 90) {
            return {
                type: "LL (Phase A-B)",
                desc: "Physical A-B Line Short. Note: Kaggle dataset trained specifically on Phase B-C.",
                guidance: "💡 Physical LL (Phase A-B) short! Note: The Kaggle dataset was trained on Phase B-C. For 100% ANN detection, test Phase B-C or click the '🟠 LL Fault' preset."
            };
        }
        if (Math.abs(Ib) > 200 && Math.abs(Ic) > 200 && Math.abs(Ia) < 90) {
            return {
                type: "LL (Phase B-C)",
                desc: "Phase B to Phase C Line Short Circuit (Matches Kaggle benchmark)",
                guidance: "✔ Standard Line-to-Line fault on Phases B & C (100% ANN match)."
            };
        }
        if (Math.abs(Ia) > 200 && Math.abs(Ic) > 200 && Math.abs(Ib) < 90) {
            return {
                type: "LL (Phase A-C)",
                desc: "Physical A-C Line Short. Note: Kaggle dataset trained on Phase B-C.",
                guidance: "💡 Physical LL (Phase A-C) short! Note: Kaggle dataset only trained on B-C."
            };
        }
        if (Math.abs(Ia) > 200 && Math.abs(Ib) > 200 && Math.abs(Ic) > 200) {
            return {
                type: "3-Phase Symmetrical (LLL)",
                desc: "Balanced 3-Phase Symmetrical Short Circuit (Ia + Ib + Ic = 0)",
                guidance: "⚡ Balanced 3-Phase Symmetrical Short Circuit across all 3 conductors."
            };
        }
    } else {
        // Ground return involved
        if (Math.abs(Ia) > 200 && Math.abs(Ib) < 90 && Math.abs(Ic) < 90) {
            return {
                type: "LG (Phase A-G)",
                desc: "Single Line-to-Ground on Phase A (Matches Kaggle benchmark)",
                guidance: "✔ Standard Single Line-to-Ground fault on Phase A."
            };
        }
        if (Math.abs(Ia) > 200 && Math.abs(Ib) > 200 && Math.abs(Ic) < 90) {
            return {
                type: "LLG (Phase A-B-G)",
                desc: "Double Line-to-Ground on Phases A & B (Matches Kaggle benchmark)",
                guidance: "✔ Standard Double Line-to-Ground fault on Phases A & B."
            };
        }
        if (Math.abs(Ia) > 200 && Math.abs(Ib) > 200 && Math.abs(Ic) > 200) {
            return {
                type: "LLLG (3-Phase with Ground)",
                desc: "3-Phase Short Circuit with Ground Return (Neutral path energized)",
                guidance: "⚡ Symmetrical 3-Phase fault with ground connection."
            };
        }
    }

    return {
        type: "Custom Line Variation",
        desc: `Injected Line Parameters: Ia=${Ia}A, Ib=${Ib}A, Ic=${Ic}A`,
        guidance: `Line currents: Ia=${Ia}A, Ib=${Ib}A, Ic=${Ic}A.`
    };
}

// Predict and update all UI elements
async function predictAndUpdate(animate = true) {
    // 1. Synchronize state.currentInputs from DOM inputs/sliders immediately
    FEATURE_KEYS.forEach(f => {
        const numInput = document.getElementById(`input-${f}`);
        const slider = document.getElementById(`slider-${f}`);
        if (numInput && numInput.value !== "" && !isNaN(parseFloat(numInput.value))) {
            state.currentInputs[f] = parseFloat(numInput.value);
        } else if (slider && !isNaN(parseFloat(slider.value))) {
            state.currentInputs[f] = parseFloat(slider.value);
        }
    });

    window.CURRENT_INPUTS = state.currentInputs;

    // Infer physical fault state from current line parameters
    const physical = inferPhysicalFault(state.currentInputs);
    if (!state.currentSource || state.currentSource.includes("Manual") || state.currentSource.includes("Live")) {
        state.currentActualFault = physical.type;
        state.currentActualDesc = physical.desc;
    }

    let result = null;
    let latencyMs = 0.45;

    // Try server endpoint if online (saves to MongoDB automatically), else client-side
    if (state.serverOnline) {
        try {
            const t0 = performance.now();
            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    ...state.currentInputs,
                    model_type: state.activeModelKey,
                    actual_fault: state.currentActualFault,
                    source: state.currentSource || 'Dashboard',
                    save_to_db: true
                })
            });
            if (res.ok) {
                result = await res.json();
                latencyMs = result.latency_ms || (performance.now() - t0).toFixed(2);
                // Update MongoDB counter badge
                if (result.document_count !== undefined) updateMongoCounter(result.document_count, result.mongodb_connected);
            }
        } catch (e) {
            console.warn("Server inference error, falling back to client-side:", e);
        }
    }

    if (!result) {
        const t0 = performance.now();
        result = runClientSideInference(state.currentInputs, state.activeModelKey);
        latencyMs = +(performance.now() - t0).toFixed(2);
        // Even in client-side mode, try to save to MongoDB via server
        if (state.serverOnline) {
            saveToMongoBackground(result);
        }
    }

    if (!result) return;

    // 1. Update Actual vs Predicted Verification Banner
    const actualTypeEl = document.getElementById('actual-fault-type');
    const actualDescEl = document.getElementById('actual-fault-desc');
    const actualSourceEl = document.getElementById('actual-source-badge');
    const validationBadge = document.getElementById('validation-match-badge');
    const relayActionEl = document.getElementById('relay-action-text');

    if (actualTypeEl) actualTypeEl.textContent = state.currentActualFault;
    if (actualDescEl) actualDescEl.textContent = state.currentActualDesc;
    if (actualSourceEl) actualSourceEl.textContent = state.currentSource;

    const isHealthy = result.predicted_fault.includes("No Fault");

    // Match logic
    if (validationBadge) {
        if (state.currentSource.includes("Manual")) {
            validationBadge.className = 'badge badge-amber text-xs font-bold py-1 px-2.5';
            validationBadge.textContent = '🔍 CUSTOM VARIATION INFERRED';
        } else if (result.predicted_fault === state.currentActualFault) {
            validationBadge.className = 'badge badge-green text-xs font-bold py-1 px-2.5';
            validationBadge.textContent = '✔ 100% ACCURATE MATCH (Correct Trip Command)';
        } else {
            validationBadge.className = 'badge badge-amber text-xs font-bold py-1 px-2.5';
            validationBadge.textContent = `⚠ MODEL DECISION: ${result.predicted_fault}`;
        }
    }

    if (relayActionEl) {
        if (isHealthy) {
            relayActionEl.className = 'text-emerald-400 font-semibold';
            relayActionEl.textContent = 'GRID NORMAL (NO TRIP)';
        } else {
            relayActionEl.className = 'text-rose-400 font-bold';
            relayActionEl.textContent = 'TRIP COMMAND ISSUED';
        }
    }

    // 2. Update Current Variation Indicator
    const currentIndicator = document.getElementById('current-state-indicator');
    if (currentIndicator) {
        const maxI = Math.max(Math.abs(state.currentInputs.Ia || 0), Math.abs(state.currentInputs.Ib || 0), Math.abs(state.currentInputs.Ic || 0));
        if (maxI > 120) {
            currentIndicator.className = 'text-xs font-bold text-rose-400 bg-slate-900/90 py-1.5 px-3 rounded-lg border border-slate-800';
            currentIndicator.innerHTML = `<span>⚠ ${physical.type} DETECTED (${maxI.toFixed(1)} A)</span><div class="text-[11px] text-slate-300 font-normal mt-0.5">${physical.guidance}</div>`;
        } else {
            currentIndicator.className = 'text-xs font-bold text-emerald-400 bg-slate-900/90 py-1.5 px-3 rounded-lg border border-slate-800';
            currentIndicator.innerHTML = `<span>🟢 NORMAL LOAD CURRENT (${maxI.toFixed(1)} A) -> SAFE CONTINUOUS</span>`;
        }
    }

    // 3. Update Result Badges
    const faultLabelEl = document.getElementById('result-fault-type');
    const confEl = document.getElementById('result-confidence');
    const latencyEl = document.getElementById('result-latency');

    if (faultLabelEl) {
        faultLabelEl.textContent = result.predicted_fault;
        faultLabelEl.className = `text-lg font-extrabold flex items-center gap-2 ${isHealthy ? 'text-emerald-400' : 'text-rose-400'}`;
    }
    if (confEl) confEl.textContent = `${result.confidence}% Conf`;
    if (latencyEl) latencyEl.textContent = `${latencyMs} ms`;

    // 4. Update Neural Network Canvas
    if (nnViz) {
        nnViz.setActivations(result.activations);
        if (animate) {
            nnViz.triggerPropagationAnimation();
        }
    }

    // 5. Update Power Systems Visualizers
    if (scope) scope.updateInputs(state.currentInputs);
    if (phasor) phasor.updateInputs(state.currentInputs);
    if (gridViz) gridViz.updateFaultState(result.predicted_fault);

    // 6. Update Step-by-Step Math Accordion
    updateMathStepper(result);

    // 7. Update Physics Card
    updatePhysicsInsightCard(result);

    // 8. Update Confusion Matrix Beside (with live active test cell highlight)
    renderConfusionMatrix(state.currentActualFault, result.predicted_fault);
}

// ─── MongoDB Helper Functions ─────────────────────────────────────────────────

function updateMongoCounter(count, connected) {
    const badge = document.getElementById('mongo-doc-count');
    if (!badge) return;
    if (connected) {
        badge.textContent = `💾 ${count} docs in MongoDB`;
        badge.className = 'badge badge-green text-xs font-mono py-1 px-2.5';
    } else {
        badge.textContent = `${count} in-memory records`;
        badge.className = 'badge badge-amber text-xs font-mono py-1 px-2.5';
    }
}

async function saveToMongoBackground(result) {
    if (!result || !state.serverOnline) return;
    try {
        const res = await fetch('/api/mongo/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                ...state.currentInputs,
                actual_fault: state.currentActualFault,
                predicted_fault: result.predicted_fault,
                confidence: result.confidence,
                model_used: state.activeModelKey,
                source: state.currentSource || 'Dashboard'
            })
        });
        if (res.ok) {
            const data = await res.json();
            updateMongoCounter(data.document_count, data.mongodb_connected);
        }
    } catch (e) {
        console.warn('MongoDB background save failed:', e);
    }
}

window.saveCurrentReadingToMongo = async function() {
    const result = runClientSideInference(state.currentInputs, state.activeModelKey);
    if (!result) return;
    const btn = document.getElementById('btn-save-mongo');
    if (btn) { btn.textContent = '⏳ Saving...'; btn.disabled = true; }
    try {
        const res = await fetch('/api/mongo/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                ...state.currentInputs,
                actual_fault: state.currentActualFault,
                predicted_fault: result.predicted_fault,
                confidence: result.confidence,
                model_used: state.activeModelKey,
                source: 'Manual Log Button'
            })
        });
        if (res.ok) {
            const data = await res.json();
            updateMongoCounter(data.document_count, data.mongodb_connected);
            if (btn) { btn.textContent = '✅ Saved!'; }
            setTimeout(() => { if (btn) { btn.textContent = '💾 Log to MongoDB'; btn.disabled = false; } }, 1500);
        } else {
            if (btn) { btn.textContent = '❌ Failed'; btn.disabled = false; }
        }
    } catch (e) {
        if (btn) { btn.textContent = '❌ Server Offline'; btn.disabled = false; }
    }
};

window.fetchMongoReadings = async function() {
    if (!state.serverOnline) { alert('Server is offline. Run server.py first.'); return; }
    try {
        const res = await fetch('/api/mongo/readings');
        if (!res.ok) return;
        const docs = await res.json();
        updateMongoCounter(docs.length, true);
        const docEl = document.getElementById('mongo-doc-viewer');
        if (docEl) {
            docEl.textContent = docs.length > 0
                ? JSON.stringify(docs.slice(0, 5), null, 2) + (docs.length > 5 ? `\n\n... and ${docs.length - 5} more documents` : '')
                : '{ "status": "No documents found in fault_db.readings" }';
        }
        // Rebuild table from real Mongo docs
        if (docs.length > 0) {
            state.telemetryRecords = docs.map(d => ({
                timestamp: d.timestamp,
                Ia: d.Ia, Ib: d.Ib, Ic: d.Ic,
                Va: d.Va, Vb: d.Vb, Vc: d.Vc,
                actual_fault: d.actual_fault,
                predicted_fault: d.predicted_fault,
                confidence: d.confidence,
                inference_ms: d.inference_ms || '-',
                status: d.status || (d.actual_fault === d.predicted_fault ? 'MATCH' : 'MISMATCH'),
                db_storage: 'MongoDB [fault_db.readings]'
            }));
            renderTelemetryTable();
        }
    } catch (e) {
        console.warn('Fetch readings failed:', e);
    }
};

window.clearMongoCollection = async function() {
    if (!confirm('Delete ALL documents from fault_db.readings?')) return;
    try {
        const res = await fetch('/api/mongo/clear', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
        if (res.ok) {
            const data = await res.json();
            updateMongoCounter(0, true);
            alert(`✅ Cleared ${data.deleted_count} documents from MongoDB.`);
            const docEl = document.getElementById('mongo-doc-viewer');
            if (docEl) docEl.textContent = '{ "status": "Collection cleared." }';
            state.telemetryRecords = [];
            renderTelemetryTable();
        }
    } catch (e) { console.warn('Clear failed:', e); }
};

function updateMathStepper(result) {
    const step1El = document.getElementById('math-step-1');
    const step2El = document.getElementById('math-step-2');
    const step3El = document.getElementById('math-step-3');
    const step4El = document.getElementById('math-step-4');

    if (step1El && result.activations && result.activations.inputs_scaled) {
        step1El.innerHTML = `
            <code>z = [${result.activations.inputs_scaled.join(', ')}]</code>
            <p class="text-xs text-slate-400 mt-1">Mean-centered & divided by feature standard deviation.</p>
        `;
    }
    if (step2El && result.activations && result.activations.layer1_a) {
        const activeCount = result.activations.layer1_a.filter(v => v > 0).length;
        step2El.innerHTML = `
            <div class="flex justify-between items-center">
                <span class="text-sm">Active Neurons (a > 0): <strong class="text-emerald">${activeCount} / ${result.activations.layer1_a.length}</strong></span>
                <span class="text-xs text-slate-400">ReLU Filter</span>
            </div>
        `;
    }
    if (step3El && result.activations && result.activations.layer2_a) {
        const activeCount = result.activations.layer2_a.filter(v => v > 0).length;
        step3El.innerHTML = `
            <div class="flex justify-between items-center">
                <span class="text-sm">Active Neurons (a > 0): <strong class="text-emerald">${activeCount} / ${result.activations.layer2_a.length}</strong></span>
                <span class="text-xs text-slate-400">Cross-Phase Patterns</span>
            </div>
        `;
    }
    if (step4El && result.probabilities && state.modelData) {
        const modelObj = state.modelData.models[state.activeModelKey];
        const classes = modelObj ? modelObj.classes : [];
        let html = '<div class="space-y-1.5 mt-1">';
        classes.forEach((c, i) => {
            const p = result.activations.layer3_probs ? (result.activations.layer3_probs[i] * 100).toFixed(1) : 0;
            const isWin = c === result.predicted_fault;
            html += `
                <div class="flex items-center text-xs">
                    <span class="w-32 truncate ${isWin ? 'text-amber font-bold' : 'text-slate-300'}">${c}</span>
                    <div class="flex-1 bg-slate-800 rounded-full h-2 mx-2 overflow-hidden">
                        <div class="${isWin ? 'bg-amber-500' : 'bg-sky-500'} h-full rounded-full" style="width: ${p}%"></div>
                    </div>
                    <span class="w-12 text-right font-mono ${isWin ? 'text-amber font-bold' : 'text-slate-400'}">${p}%</span>
                </div>
            `;
        });
        html += '</div>';
        step4El.innerHTML = html;
    }
}

function updatePhysicsInsightCard(result) {
    const textContainer = document.getElementById('physics-explanation-content');
    if (!textContainer || !result.power_physics) return;

    const bullets = result.power_physics.electrical_explanation || [];
    let html = '<ul class="space-y-1.5">';
    bullets.forEach(b => {
        html += `<li class="flex items-start"><span class="text-sky-400 mr-2">⚡</span><span>${b}</span></li>`;
    });
    html += '</ul>';

    html += `
        <div class="grid grid-cols-2 gap-2 mt-3 pt-3 border-t border-slate-800 text-xs">
            <div class="bg-slate-900/60 p-2 rounded">
                <span class="text-slate-400">Ground Residual Current (Ig):</span>
                <strong class="${Math.abs(result.power_physics.ground_current_ig) > 20 ? 'text-crimson' : 'text-emerald'} block font-mono text-sm">
                    ${result.power_physics.ground_current_ig} A
                </strong>
            </div>
            <div class="bg-slate-900/60 p-2 rounded">
                <span class="text-slate-400">Zero-Sequence Voltage (Vg):</span>
                <strong class="text-slate-200 block font-mono text-sm">
                    ${result.power_physics.zero_sequence_voltage_vg} p.u.
                </strong>
            </div>
        </div>
    `;

    textContainer.innerHTML = html;
}

// Global function to apply a preset (called by buttons)
window.applyPreset = function(presetKey) {
    const preset = PRESETS[presetKey];
    if (!preset) return;

    state.currentActualFault = preset.badge;
    state.currentActualDesc = preset.name;
    state.currentSource = "IEEE / Preset Standard";

    // Update inputs
    FEATURE_KEYS.forEach(f => {
        state.currentInputs[f] = preset.vals[f];
        const slider = document.getElementById(`slider-${f}`);
        const numInput = document.getElementById(`input-${f}`);
        if (slider) slider.value = preset.vals[f];
        if (numInput) numInput.value = preset.vals[f];
    });

    // Update active button styling
    document.querySelectorAll('.preset-btn, .btn-preset').forEach(btn => {
        btn.classList.remove('active-preset', 'ring-2', 'ring-sky-400', 'bg-sky-950/60');
    });
    const clickedBtn = document.getElementById(`preset-${presetKey}`);
    if (clickedBtn) {
        clickedBtn.classList.add('active-preset');
    }

    // Trigger prediction with animated signal pulse
    predictAndUpdate(true);
};

// Global function to trigger prediction explicitly
window.checkFaultNow = function() {
    predictAndUpdate(true);
};

// Load Kaggle Sample by Index (Dropdown)
window.loadSampleByIndex = function(index) {
    if (!state.modelData || !state.modelData.test_samples) return;
    const idx = parseInt(index, 10);
    const sample = state.modelData.test_samples[idx];
    if (!sample) return;

    state.currentActualFault = sample.actual_fault;
    state.currentActualDesc = `Kaggle classData.csv Test Row #${idx} Ground Truth`;
    state.currentSource = "Kaggle classData.csv";

    FEATURE_KEYS.forEach(f => {
        state.currentInputs[f] = sample[f];
        const slider = document.getElementById(`slider-${f}`);
        const numInput = document.getElementById(`input-${f}`);
        if (slider) slider.value = sample[f];
        if (numInput) numInput.value = sample[f];
    });

    document.querySelectorAll('.btn-preset').forEach(btn => {
        btn.classList.remove('ring-2', 'ring-sky-400', 'bg-sky-950/60');
    });

    predictAndUpdate(true);
};

// Load Random Kaggle Test Sample
window.loadRandomSample = function() {
    if (!state.modelData || !state.modelData.test_samples || !state.modelData.test_samples.length) return;
    state.sampleIndex = (state.sampleIndex + 1) % state.modelData.test_samples.length;
    const sample = state.modelData.test_samples[state.sampleIndex];

    state.currentActualFault = sample.actual_fault;
    state.currentActualDesc = `Kaggle classData.csv Test Row #${state.sampleIndex} Ground Truth`;
    state.currentSource = "Kaggle classData.csv";

    FEATURE_KEYS.forEach(f => {
        state.currentInputs[f] = sample[f];
        const slider = document.getElementById(`slider-${f}`);
        const numInput = document.getElementById(`input-${f}`);
        if (slider) slider.value = sample[f];
        if (numInput) numInput.value = sample[f];
    });

    document.querySelectorAll('.btn-preset').forEach(btn => {
        btn.classList.remove('ring-2', 'ring-sky-400', 'bg-sky-950/60');
    });

    predictAndUpdate(true);
};

// Interactive Current Surge Injection Demo
window.injectCurrentSurge = function() {
    // If Phase A current magnitude is currently small (< 200A), surge to -750A
    if (Math.abs(state.currentInputs.Ia) < 200) {
        state.currentActualFault = "LG (Injected Surge)";
        state.currentActualDesc = "Injected Severe Phase A Current Surge (-750 A)";
        state.currentSource = "Live Surge Test";
        state.currentInputs.Ia = -750.0;
        state.currentInputs.Va = 0.12;
    } else {
        // Toggle back to normal nominal load
        state.currentActualFault = "No Fault";
        state.currentActualDesc = "Restored Healthy Continuous Load (-15.2 A)";
        state.currentSource = "Live Surge Test";
        state.currentInputs.Ia = -15.2;
        state.currentInputs.Va = 0.58;
    }

    FEATURE_KEYS.forEach(f => {
        const slider = document.getElementById(`slider-${f}`);
        const numInput = document.getElementById(`input-${f}`);
        if (slider) slider.value = state.currentInputs[f];
        if (numInput) numInput.value = state.currentInputs[f];
    });

    predictAndUpdate(true);
};

// Model Switcher
window.switchModel = function(modelKey) {
    state.activeModelKey = modelKey;
    document.querySelectorAll('.model-select-input').forEach(sel => {
        sel.value = modelKey;
    });
    const modelSelector = document.getElementById('model-select');
    if (modelSelector) modelSelector.value = modelKey;
    setupModelArchitecture();
    predictAndUpdate(true);
    renderPerformanceTable();
};

function setupModelArchitecture() {
    if (!state.modelData || !nnViz) return;
    const modelObj = state.modelData.models[state.activeModelKey] || state.modelData.models['baseline'];
    if (!modelObj) return;

    nnViz.setModelArchitecture(
        modelObj.architecture,
        modelObj.classes,
        state.modelData.metadata.features
    );
}

// Navigation Tab Switching
window.switchTab = function(tabId) {
    document.querySelectorAll('.nav-tab-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tabId);
    });
    document.querySelectorAll('.tab-section').forEach(sec => {
        sec.classList.toggle('hidden', sec.id !== tabId);
    });

    setTimeout(() => {
        if (nnViz) nnViz.resize();
        if (scope) scope.resize();
        if (phasor) phasor.resize();
    }, 60);
};

// Slide Deck Navigation
window.nextSlide = function() {
    if (state.activeSlide < state.totalSlides) goToSlide(state.activeSlide + 1);
};
window.prevSlide = function() {
    if (state.activeSlide > 1) goToSlide(state.activeSlide - 1);
};
window.goToSlide = function(num) {
    state.activeSlide = num;
    document.querySelectorAll('.slide-page').forEach((el, idx) => {
        el.style.display = (idx + 1 === num) ? 'block' : 'none';
    });
    const counterEl = document.getElementById('slide-counter-badge');
    if (counterEl) counterEl.textContent = `Slide ${num} of ${state.totalSlides}`;

    document.querySelectorAll('.slide-nav-pill').forEach((pill, idx) => {
        pill.className = (idx + 1 === num) ? 'slide-nav-pill active' : 'slide-nav-pill';
    });
};

// Helper to match class names between actual label and matrix row
function isMatchingClass(cellClass, targetClass) {
    if (!cellClass || !targetClass) return false;
    if (cellClass === targetClass) return true;
    const norm = (s) => String(s).toLowerCase().replace(/[^a-z0-9]/g, '');
    const c1 = norm(cellClass);
    const c2 = norm(targetClass);
    if (c1 === c2) return true;
    if (c1.startsWith(c2) || c2.startsWith(c1)) return true;
    // Symmetrical fault matching
    if ((cellClass.includes("LLL") || cellClass.includes("3-Phase")) && 
        (targetClass.includes("LLL") || targetClass.includes("3-Phase"))) {
        return true;
    }
    return false;
}

// Render Interactive Confusion Matrix Beside the Model View
function renderConfusionMatrix(activeTrueClass, activePredClass) {
    const matrixContainer = document.getElementById('confusion-matrix-container');
    if (!matrixContainer || !state.modelData) return;

    const modelObj = state.modelData.models[state.activeModelKey] || state.modelData.models['protective_standard'] || state.modelData.models['baseline'];
    if (!modelObj || !modelObj.confusion_matrix) return;

    const classes = modelObj.classes;
    const matrix = modelObj.confusion_matrix;

    const getShortLabel = (c) => {
        if (c.includes("LG")) return "LG";
        if (c.includes("LLG")) return "LLG";
        if (c.includes("LLLG")) return "LLLG";
        if (c.includes("LLL")) return "LLL";
        if (c.includes("LL")) return "LL";
        if (c.includes("3-Phase") || c.includes("Symm")) return "3-Ph";
        if (c.includes("No Fault")) return "Normal";
        return c;
    };

    let totalSamples = 0;
    let totalCorrect = 0;

    matrix.forEach((row, rIdx) => {
        row.forEach((count, cIdx) => {
            totalSamples += count;
            if (rIdx === cIdx) totalCorrect += count;
        });
    });

    const modelAcc = modelObj.accuracy || ((totalCorrect / totalSamples) * 100).toFixed(2);

    let tableHtml = '<div class="overflow-x-auto"><table class="w-full text-xs text-center border-collapse font-mono">';
    tableHtml += '<thead><tr class="border-b border-slate-800 text-[10px]">';
    tableHtml += '<th class="py-1.5 px-2 text-left text-slate-400 font-semibold uppercase tracking-wider">True \\ Pred</th>';
    classes.forEach(c => {
        const short = getShortLabel(c);
        tableHtml += `<th class="py-1.5 px-1 text-sky-400 font-bold truncate max-w-[60px]" title="Predicted: ${c}">${short}</th>`;
    });
    tableHtml += '<th class="py-1.5 px-2 text-emerald-400 font-bold text-right">Recall</th>';
    tableHtml += '</tr></thead><tbody>';

    matrix.forEach((row, rIdx) => {
        const trueClass = classes[rIdx];
        const rowSum = row.reduce((a, b) => a + b, 0);
        const correctCount = row[rIdx];
        const recallPct = rowSum > 0 ? ((correctCount / rowSum) * 100).toFixed(1) : 0;
        const shortTrue = getShortLabel(trueClass);

        tableHtml += `<tr class="border-b border-slate-800/60 hover:bg-slate-900/40 transition">`;
        tableHtml += `<td class="py-1.5 px-2 text-left text-slate-300 font-bold text-xs" title="True Class: ${trueClass}">${shortTrue}</td>`;

        row.forEach((count, cIdx) => {
            const predClass = classes[cIdx];
            const isDiag = rIdx === cIdx;
            const pct = rowSum > 0 ? ((count / rowSum) * 100).toFixed(1) : 0;

            const isCurrent = isMatchingClass(trueClass, activeTrueClass) && isMatchingClass(predClass, activePredClass);

            let bg = 'bg-slate-900/30';
            let textCol = count === 0 ? 'text-slate-600' : 'text-slate-400';
            let extraClass = '';

            if (isDiag && count > 0) {
                bg = 'bg-emerald-950/70 border border-emerald-500/30';
                textCol = 'text-emerald-300 font-bold';
            } else if (!isDiag && count > 0) {
                bg = 'bg-rose-950/70 border border-rose-500/30';
                textCol = 'text-rose-300 font-bold';
            }

            if (isCurrent) {
                extraClass = 'cm-active-cell';
            }

            tableHtml += `
                <td class="p-1.5 ${bg} ${textCol} ${extraClass} rounded transition-all cursor-default relative matrix-cell" 
                    title="True: ${trueClass} | Pred: ${predClass} (${count} samples, ${pct}%)">
                    <div class="text-[12px] leading-tight">${count}</div>
                    <div class="text-[8.5px] opacity-75 font-sans">${pct}%</div>
                    ${isCurrent ? '<span class="absolute -top-1 -right-1 w-2 h-2 rounded-full bg-amber-400 animate-ping"></span>' : ''}
                </td>
            `;
        });

        tableHtml += `<td class="py-1.5 px-2 text-right font-mono font-bold text-xs ${recallPct > 95 ? 'text-emerald-400' : 'text-amber-400'}">${recallPct}%</td>`;
        tableHtml += '</tr>';
    });

    tableHtml += '</tbody></table></div>';

    // Summary strip below matrix
    tableHtml += `
        <div class="mt-2.5 pt-2 border-t border-slate-800 flex items-center justify-between text-xs">
            <div class="flex items-center gap-1.5">
                <span class="text-slate-400">Accuracy:</span>
                <span class="font-extrabold text-sm ${modelAcc > 95 ? 'text-emerald-400' : 'text-amber-400'}">${modelAcc}%</span>
                <span class="text-[10px] text-slate-500 font-mono">(${totalCorrect}/${totalSamples})</span>
            </div>
            <div class="text-[10px] text-slate-400 flex items-center gap-2">
                <span class="inline-flex items-center gap-1"><span class="w-2 h-2 rounded bg-emerald-500"></span> Correct</span>
                <span class="inline-flex items-center gap-1"><span class="w-2 h-2 rounded bg-rose-500"></span> Error</span>
                <span class="inline-flex items-center gap-1"><span class="w-2 h-2 rounded bg-amber-400"></span> Active</span>
            </div>
        </div>
    `;

    matrixContainer.innerHTML = tableHtml;
}

// Power Visualization Toggle (Oscilloscope vs Phasor)
window.togglePowerViz = function(mode) {
    const scopeCard = document.getElementById('scope-container');
    const phasorCard = document.getElementById('phasor-container');
    const btnScope = document.getElementById('btn-tab-scope');
    const btnPhasor = document.getElementById('btn-tab-phasor');

    if (mode === 'scope') {
        if (scopeCard) scopeCard.style.display = 'block';
        if (phasorCard) phasorCard.style.display = 'none';
        if (btnScope) btnScope.className = 'tab-pill active';
        if (btnPhasor) btnPhasor.className = 'tab-pill';
        if (scope) scope.resize();
    } else {
        if (scopeCard) scopeCard.style.display = 'none';
        if (phasorCard) phasorCard.style.display = 'flex';
        if (btnScope) btnScope.className = 'tab-pill';
        if (btnPhasor) btnPhasor.className = 'tab-pill active';
        if (phasor) phasor.resize();
    }
};

function renderPerformanceTable() {
    const perfContainer = document.getElementById('model-comparison-tbody');
    if (!perfContainer || !state.modelData) return;

    const m = state.modelData.models;
    const modelsList = [
        { key: 'baseline', name: 'Baseline 6-Class MLP (Current Proposal)', acc: m.baseline.accuracy, notes: 'Struggles with LLL vs LLLG symmetry cancellation' },
        { key: 'enhanced', name: 'Enhanced Deep 6-Class MLP', acc: m.enhanced.accuracy, notes: 'Deeper architecture (64-32) with adaptive Adam rate' },
        { key: 'protective_standard', name: '5-Class IEEE Protective Relay Standard', acc: m.protective_standard.accuracy, notes: 'Symmetrical 3-Phase grouping. Solves Kirchhoff ground zero-current issue!' },
    ];

    let html = '';
    modelsList.forEach(mItem => {
        const isCurrent = mItem.key === state.activeModelKey;
        html += `
            <tr class="${isCurrent ? 'bg-sky-950/40 border-l-4 border-sky-400' : 'hover:bg-slate-800/40'} border-b border-slate-800/80">
                <td class="py-3 px-4 font-medium text-slate-200">
                    ${mItem.name}
                    ${isCurrent ? '<span class="ml-2 text-xs text-sky-400 font-semibold">(Active)</span>' : ''}
                </td>
                <td class="py-3 px-4">
                    <strong class="${mItem.acc > 95 ? 'text-emerald-400' : 'text-amber-400'} font-mono text-base">
                        ${mItem.acc}%
                    </strong>
                </td>
                <td class="py-3 px-4 text-xs text-slate-400">${mItem.notes}</td>
                <td class="py-3 px-4">
                    <button class="btn-xs ${isCurrent ? 'btn-disabled' : 'btn-outline'}" onclick="window.switchModel('${mItem.key}')">
                        ${isCurrent ? 'Selected' : 'Switch Model'}
                    </button>
                </td>
            </tr>
        `;
    });

    perfContainer.innerHTML = html;
}

// Telemetry Streaming
window.toggleStream = function() {
    if (state.isStreaming) stopStream();
    else startStream();
};

function startStream() {
    state.isStreaming = true;
    const btn = document.getElementById('btn-toggle-stream');
    if (btn) {
        btn.innerHTML = '⏸ Pause Streaming';
        btn.className = 'btn-secondary';
    }

    const streamBadge = document.getElementById('stream-live-indicator');
    if (streamBadge) streamBadge.style.display = 'inline-flex';

    state.streamInterval = setInterval(async () => {
        await fetchNextStreamRecord();
    }, state.streamSpeedMs);
}

function stopStream() {
    state.isStreaming = false;
    clearInterval(state.streamInterval);
    const btn = document.getElementById('btn-toggle-stream');
    if (btn) {
        btn.innerHTML = '▶ Start Telemetry Streaming';
        btn.className = 'btn-primary';
    }

    const streamBadge = document.getElementById('stream-live-indicator');
    if (streamBadge) streamBadge.style.display = 'none';
}

async function fetchNextStreamRecord() {
    let record = null;
    if (state.serverOnline) {
        try {
            const res = await fetch('/api/stream/next');
            if (res.ok) {
                const data = await res.json();
                record = data.record;
                // Update MongoDB counter from server response
                if (data.document_count !== undefined) {
                    updateMongoCounter(data.document_count, data.mongodb_connected);
                }
            }
        } catch (e) {
            console.warn("Stream server fetch failed:", e);
        }
    }

    if (!record && state.modelData && state.modelData.test_samples && state.modelData.test_samples.length) {
        const streamModeSelect = document.getElementById('stream-mode-select');
        const mode = streamModeSelect ? streamModeSelect.value : 'random';
        let sample;

        if (mode === 'random') {
            const randIdx = Math.floor(Math.random() * state.modelData.test_samples.length);
            sample = state.modelData.test_samples[randIdx];
        } else if (mode === 'faults_only') {
            const faults = state.modelData.test_samples.filter(s => !s.actual_fault.includes('No Fault'));
            sample = faults[Math.floor(Math.random() * faults.length)];
        } else {
            // Interleaved sequential (alternates: LG -> LL -> LLG -> LLL -> LLLG -> No Fault...)
            state.sampleIndex = (state.sampleIndex + 1) % state.modelData.test_samples.length;
            sample = state.modelData.test_samples[state.sampleIndex];
        }

        const t0 = performance.now();
        const pred = runClientSideInference(sample, state.activeModelKey);
        const lat = +(performance.now() - t0).toFixed(2);

        record = {
            timestamp: new Date().toLocaleTimeString(),
            Ia: sample.Ia,
            Ib: sample.Ib,
            Ic: sample.Ic,
            Va: sample.Va,
            Vb: sample.Vb,
            Vc: sample.Vc,
            actual_fault: sample.actual_fault,
            predicted_fault: pred.predicted_fault,
            confidence: pred.confidence,
            inference_ms: lat,
            status: sample.actual_fault === pred.predicted_fault ? "MATCH" : "MISMATCH",
            db_storage: "MongoDB [fault_db.readings]"
        };
    }

    if (!record) return;

    // Synchronize top verification banner
    state.currentActualFault = record.actual_fault;
    state.currentActualDesc = `Live Telemetry Streaming Reading`;
    state.currentSource = record.db_storage;

    // Apply inputs to current view
    FEATURE_KEYS.forEach(f => {
        state.currentInputs[f] = record[f];
        const slider = document.getElementById(`slider-${f}`);
        const numInput = document.getElementById(`input-${f}`);
        if (slider) slider.value = record[f];
        if (numInput) numInput.value = record[f];
    });
    predictAndUpdate(false);

    // Prepend to telemetry table
    state.telemetryRecords.unshift(record);
    if (state.telemetryRecords.length > 25) state.telemetryRecords.pop();

    renderTelemetryTable();
    renderMongoDocument(record);
}

function renderTelemetryTable() {
    const streamTableBody = document.getElementById('telemetry-table-body');
    if (!streamTableBody) return;
    let html = '';
    state.telemetryRecords.forEach((r, idx) => {
        const isMatch = r.status === "MATCH";
        const isHealthy = r.predicted_fault.includes("No Fault");
        html += `
            <tr class="${idx === 0 ? 'bg-sky-950/30 font-medium' : ''} border-b border-slate-800/60">
                <td class="py-2 px-3 font-mono text-xs text-slate-400">${r.timestamp.split('T')[1] ? r.timestamp.split('T')[1].slice(0,8) : r.timestamp}</td>
                <td class="py-2 px-3 font-mono text-xs">${r.Ia}</td>
                <td class="py-2 px-3 font-mono text-xs">${r.Ib}</td>
                <td class="py-2 px-3 font-mono text-xs">${r.Ic}</td>
                <td class="py-2 px-3 font-mono text-xs">${r.Va}</td>
                <td class="py-2 px-3 font-mono text-xs">${r.Vb}</td>
                <td class="py-2 px-3 font-mono text-xs">${r.Vc}</td>
                <td class="py-2 px-3">
                    <span class="badge ${isHealthy ? 'badge-green' : 'badge-red'}">${r.predicted_fault}</span>
                </td>
                <td class="py-2 px-3 font-mono text-xs text-slate-400">${r.actual_fault}</td>
                <td class="py-2 px-3 font-mono text-xs text-emerald">${r.inference_ms}ms</td>
                <td class="py-2 px-3">
                    <span class="badge ${isMatch ? 'badge-green' : 'badge-amber'}">${r.status}</span>
                </td>
            </tr>
        `;
    });
    streamTableBody.innerHTML = html;
}

function renderMongoDocument(record) {
    const docEl = document.getElementById('mongo-doc-viewer');
    if (!docEl) return;
    const mongoDoc = {
        "_id": record._id ? `ObjectId("${record._id}")` : `ObjectId("${Date.now().toString(16).padStart(24,'0')}")`,
        "timestamp": record.timestamp || new Date().toISOString(),
        "telemetry": {
            "Ia": record.Ia, "Ib": record.Ib, "Ic": record.Ic,
            "Va": record.Va, "Vb": record.Vb, "Vc": record.Vc
        },
        "prediction": {
            "fault_type": record.predicted_fault,
            "confidence": record.confidence + "%",
            "inference_time_ms": record.inference_ms
        },
        "actual_fault": record.actual_fault,
        "status": record.status,
        "grid_status": record.predicted_fault.includes("No Fault") ? "ENERGIZED" : "TRIP_COMMAND_ISSUED",
        "db_collection": "fault_db.readings",
        "db_storage": record.db_storage || "MongoDB [fault_db.readings]"
    };
    docEl.textContent = JSON.stringify(mongoDoc, null, 2);
}

// Master Initialization Function
function initApp() {
    // 1. Initialize Visualizers
    nnViz = new NeuralNetworkVisualizer('nn-canvas');
    scope = new ThreePhaseOscilloscope('scope-canvas');
    phasor = new PhasorDiagram('phasor-canvas');
    gridViz = new TransmissionGridVisualizer('transmission-grid-card');

    // 2. Wire Slider & Number Inputs
    FEATURE_KEYS.forEach(f => {
        const slider = document.getElementById(`slider-${f}`);
        const numInput = document.getElementById(`input-${f}`);

        if (slider && numInput) {
            const handleValChange = (val, fromSlider) => {
                if (isNaN(val)) return;
                state.currentInputs[f] = val;
                if (fromSlider) {
                    numInput.value = val;
                } else {
                    slider.value = val;
                }
                const physical = inferPhysicalFault(state.currentInputs);
                state.currentActualFault = physical.type;
                state.currentActualDesc = physical.desc;
                state.currentSource = "Manual Live Injection";
                predictAndUpdate(false);
            };

            slider.addEventListener('input', (e) => handleValChange(parseFloat(e.target.value), true));
            slider.addEventListener('change', (e) => handleValChange(parseFloat(e.target.value), true));

            numInput.addEventListener('input', (e) => {
                const v = parseFloat(e.target.value);
                if (!isNaN(v)) handleValChange(v, false);
            });
            numInput.addEventListener('change', (e) => {
                const v = parseFloat(e.target.value);
                if (!isNaN(v)) handleValChange(v, false);
            });
        }
    });

    // 3. Wire Model Selector Dropdown
    const modelSelector = document.getElementById('model-select');
    if (modelSelector) {
        modelSelector.addEventListener('change', (e) => {
            window.switchModel(e.target.value);
        });
    }

    // 4. Wire Oscilloscope Current/Voltage Toggle
    const scopeCurrentBtn = document.getElementById('btn-scope-current');
    const scopeVoltageBtn = document.getElementById('btn-scope-voltage');
    if (scopeCurrentBtn && scopeVoltageBtn) {
        scopeCurrentBtn.addEventListener('click', () => {
            scope.setMode('current');
            scopeCurrentBtn.className = 'btn-xs tab-pill active';
            scopeVoltageBtn.className = 'btn-xs tab-pill';
        });
        scopeVoltageBtn.addEventListener('click', () => {
            scope.setMode('voltage');
            scopeVoltageBtn.className = 'btn-xs tab-pill active';
            scopeCurrentBtn.className = 'btn-xs tab-pill';
        });
    }

    // 5. Wire Streaming Controls
    const streamToggleBtn = document.getElementById('btn-toggle-stream');
    if (streamToggleBtn) {
        streamToggleBtn.addEventListener('click', window.toggleStream);
    }

    const streamSpeedSelect = document.getElementById('stream-speed-select');
    if (streamSpeedSelect) {
        streamSpeedSelect.addEventListener('change', (e) => {
            state.streamSpeedMs = parseInt(e.target.value, 10);
            if (state.isStreaming) {
                stopStream();
                startStream();
            }
        });
    }

    // 6. Setup Initial Model Architecture & Data
    if (state.modelData) {
        document.querySelectorAll('.model-select-input').forEach(sel => {
            sel.value = state.activeModelKey;
        });
        setupModelArchitecture();
        renderConfusionMatrix();
        renderPerformanceTable();
    }

    // 7. Check server status in background
    fetch('/api/status', { cache: 'no-store' })
        .then(r => r.json())
        .then(data => {
            state.serverOnline = true;
            const dbBadge = document.getElementById('db-status-badge');
            if (dbBadge) {
                if (data.mongodb_connected) {
                    dbBadge.className = 'status-pill online';
                    dbBadge.innerHTML = '● MongoDB: Live (Port 27017)';
                } else {
                    dbBadge.className = 'status-pill simulated';
                    dbBadge.innerHTML = '● Real-Time DB: In-Memory / Fallback Store';
                }
            }
        })
        .catch(() => {
            state.serverOnline = false;
            const dbBadge = document.getElementById('db-status-badge');
            if (dbBadge) {
                dbBadge.className = 'status-pill client-only';
                dbBadge.innerHTML = '● Standalone Client-Side AI Engine (Active)';
            }
        });

    // 8. Apply Initial LG Fault Preset
    window.applyPreset('lg');
}

// Ensure initApp runs regardless of whether DOM is already loaded or loading
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initApp);
} else {
    initApp();
}

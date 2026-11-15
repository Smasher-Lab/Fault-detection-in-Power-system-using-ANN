/**
 * Neural Network Canvas Visualizer with Animated Synaptic Signal Propagation
 * Visualizes the Multilayer Perceptron (MLP) architecture, node activations (ReLU),
 * synaptic weight intensities, and step-by-step forward propagation pulses.
 */

class NeuralNetworkVisualizer {
    constructor(canvasId, options = {}) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) return;
        this.ctx = this.canvas.getContext('2d');
        this.options = Object.assign({
            nodeRadius: 14,
            pulseSpeed: 0.03,
            maxHiddenRender: 16 // maximum nodes to display per hidden layer for clean visual clarity
        }, options);

        this.particles = [];
        this.currentStep = 0; // 0: Idle, 1: L0->L1, 2: L1->L2, 3: L2->L3, 4: Done
        this.stepProgress = 0;
        this.isAnimating = false;
        this.hoveredNode = null;

        this.modelData = null;
        this.layerSizes = [6, 16, 12, 6];
        this.nodePositions = [];
        this.activations = null;
        this.weights = null;
        this.biases = null;
        this.classes = ["LG (A-G)", "LL (B-C)", "LLG (A-B-G)", "LLL (A-B-C)", "LLLG (A-B-C-G)", "No Fault"];
        this.inputLabels = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"];

        this.initEventListeners();
        this.resize();
        window.addEventListener('resize', () => this.resize());
        this.startRenderLoop();
    }

    resize() {
        if (!this.canvas) return;
        const rect = this.canvas.parentElement.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        this.width = rect.width;
        this.height = Math.max(360, Math.min(rect.width * 0.78, 430));
        this.canvas.width = this.width * dpr;
        this.canvas.height = this.height * dpr;
        this.canvas.style.width = `${this.width}px`;
        this.canvas.style.height = `${this.height}px`;
        this.ctx.scale(dpr, dpr);
        this.computeNodePositions();
    }

    setModelArchitecture(archData, classes, inputLabels = null) {
        if (!archData) return;
        this.modelData = archData;
        if (classes) this.classes = classes;
        if (inputLabels) this.inputLabels = inputLabels;

        // Simplify hidden layer counts for display if they are large
        const rawSizes = archData.layer_sizes;
        this.layerSizes = [
            rawSizes[0],
            Math.min(rawSizes[1], this.options.maxHiddenRender),
            Math.min(rawSizes[2], Math.round(this.options.maxHiddenRender * 0.75)),
            rawSizes[3]
        ];

        this.weights = archData.weights;
        this.biases = archData.biases;
        this.computeNodePositions();
    }

    setActivations(activations) {
        this.activations = activations;
    }

    computeNodePositions() {
        this.nodePositions = [];
        if (!this.width || !this.height) return;

        const numLayers = this.layerSizes.length;
        const leftMargin = this.width < 500 ? 55 : 75;
        const rightMargin = this.width < 500 ? 115 : 135;
        const availableX = Math.max(160, this.width - leftMargin - rightMargin);
        const layerXGap = availableX / (numLayers - 1);
        const startX = leftMargin;

        for (let l = 0; l < numLayers; l++) {
            const count = this.layerSizes[l];
            const layerNodes = [];
            const layerHeight = this.height - 84;
            const nodeYGap = count > 1 ? layerHeight / (count - 1) : 0;
            const startY = 44;

            for (let n = 0; n < count; n++) {
                layerNodes.push({
                    layer: l,
                    index: n,
                    x: startX + l * layerXGap,
                    y: startY + (count > 1 ? n * nodeYGap : layerHeight / 2),
                    radius: l === 0 || l === numLayers - 1 ? (this.width < 500 ? 12 : 14) : (this.width < 500 ? 8 : 10)
                });
            }
            this.nodePositions.push(layerNodes);
        }
    }

    initEventListeners() {
        this.canvas.addEventListener('mousemove', (e) => {
            const rect = this.canvas.getBoundingClientRect();
            const mx = e.clientX - rect.left;
            const my = e.clientY - rect.top;
            let found = null;

            for (const layer of this.nodePositions) {
                for (const node of layer) {
                    const dist = Math.hypot(node.x - mx, node.y - my);
                    if (dist < node.radius + 6) {
                        found = node;
                        break;
                    }
                }
                if (found) break;
            }
            this.hoveredNode = found;
            this.updateTooltip(found, e.clientX, e.clientY);
        });

        this.canvas.addEventListener('mouseleave', () => {
            this.hoveredNode = null;
            this.updateTooltip(null);
        });
    }

    updateTooltip(node, clientX, clientY) {
        const tooltip = document.getElementById('nn-tooltip');
        if (!tooltip) return;

        if (!node) {
            tooltip.style.opacity = '0';
            tooltip.style.pointerEvents = 'none';
            return;
        }

        const lNames = ["Input Layer", "Hidden Layer 1 (ReLU)", "Hidden Layer 2 (ReLU)", "Output Layer (Softmax)"];
        let content = `<strong>${lNames[node.layer]}</strong><br>`;

        if (node.layer === 0) {
            const feat = this.inputLabels[node.index] || `X${node.index}`;
            const scaled = this.activations && this.activations.inputs_scaled ? this.activations.inputs_scaled[node.index] : 'N/A';
            content += `Feature: <span class="text-cyan">${feat}</span><br>Normalized (z-score): <strong>${scaled}</strong>`;
        } else if (node.layer === 1) {
            const z = this.activations && this.activations.layer1_z ? this.activations.layer1_z[node.index] : 'N/A';
            const a = this.activations && this.activations.layer1_a ? this.activations.layer1_a[node.index] : 'N/A';
            content += `Neuron #${node.index + 1}<br>Weighted Sum z: <strong>${z}</strong><br>ReLU a = max(0, z): <strong class="text-emerald">${a}</strong>`;
        } else if (node.layer === 2) {
            const z = this.activations && this.activations.layer2_z ? this.activations.layer2_z[node.index] : 'N/A';
            const a = this.activations && this.activations.layer2_a ? this.activations.layer2_a[node.index] : 'N/A';
            content += `Neuron #${node.index + 1}<br>Weighted Sum z: <strong>${z}</strong><br>ReLU a = max(0, z): <strong class="text-emerald">${a}</strong>`;
        } else {
            const cls = this.classes[node.index] || `Class ${node.index}`;
            const prob = this.activations && this.activations.layer3_probs ? (this.activations.layer3_probs[node.index] * 100).toFixed(2) + '%' : 'N/A';
            content += `Class: <span class="text-amber">${cls}</span><br>Softmax Probability: <strong>${prob}</strong>`;
        }

        tooltip.innerHTML = content;
        tooltip.style.left = `${clientX + 14}px`;
        tooltip.style.top = `${clientY + 14}px`;
        tooltip.style.opacity = '1';
    }

    triggerPropagationAnimation() {
        this.particles = [];
        this.currentStep = 1;
        this.stepProgress = 0;
        this.isAnimating = true;
        this.spawnParticlesForLayer(0, 1);
    }

    spawnParticlesForLayer(srcLayerIdx, dstLayerIdx) {
        if (!this.nodePositions[srcLayerIdx] || !this.nodePositions[dstLayerIdx]) return;
        const srcNodes = this.nodePositions[srcLayerIdx];
        const dstNodes = this.nodePositions[dstLayerIdx];

        for (const src of srcNodes) {
            for (const dst of dstNodes) {
                // Determine sample weight or random intensity
                const p = {
                    srcX: src.x,
                    srcY: src.y,
                    dstX: dst.x,
                    dstY: dst.y,
                    progress: 0,
                    speed: 0.02 + Math.random() * 0.015,
                    size: 2.5 + Math.random() * 2,
                    layerFrom: srcLayerIdx,
                    layerTo: dstLayerIdx,
                    color: Math.random() > 0.4 ? 'rgba(56, 189, 248, 0.9)' : 'rgba(236, 72, 153, 0.8)'
                };
                this.particles.push(p);
            }
        }
    }

    startRenderLoop() {
        const loop = () => {
            this.render();
            requestAnimationFrame(loop);
        };
        requestAnimationFrame(loop);
    }

    render() {
        if (!this.ctx || !this.width || !this.height) return;
        this.ctx.clearRect(0, 0, this.width, this.height);

        // 1. Draw layer background headers
        this.drawLayerHeaders();

        // 2. Draw Synapses (Connections)
        this.drawSynapses();

        // 3. Update and draw animated signal pulses
        this.updateAndDrawParticles();

        // 4. Draw Neurons
        this.drawNeurons();

        // 5. Draw Output Bars / Badges
        this.drawOutputLabels();
    }

    drawLayerHeaders() {
        const headers = [
            { title: "INPUT LAYER", subtitle: "6 Normalized Measurements" },
            { title: "HIDDEN 1", subtitle: "Feature Extraction (ReLU)" },
            { title: "HIDDEN 2", subtitle: "Pattern Associator (ReLU)" },
            { title: "OUTPUT LAYER", subtitle: "Softmax Classification" }
        ];

        this.ctx.save();
        for (let l = 0; l < this.nodePositions.length; l++) {
            const nodes = this.nodePositions[l];
            if (!nodes || nodes.length === 0) continue;
            const x = nodes[0].x;
            const info = headers[l] || { title: `LAYER ${l}`, subtitle: "" };

            this.ctx.fillStyle = '#94a3b8';
            this.ctx.font = '600 11px Inter, sans-serif';
            this.ctx.textAlign = 'center';
            this.ctx.fillText(info.title, x, 24);

            this.ctx.fillStyle = '#64748b';
            this.ctx.font = '400 9px Inter, sans-serif';
            this.ctx.fillText(info.subtitle, x, 38);
        }
        this.ctx.restore();
    }

    drawSynapses() {
        if (this.nodePositions.length < 2) return;
        this.ctx.save();

        for (let l = 0; l < this.nodePositions.length - 1; l++) {
            const currentLayer = this.nodePositions[l];
            const nextLayer = this.nodePositions[l + 1];

            for (const n1 of currentLayer) {
                for (const n2 of nextLayer) {
                    let alpha = 0.06;
                    let strokeColor = 'rgba(148, 163, 184, 0.12)';

                    // If hovered node connects to this synapse, highlight it
                    if (this.hoveredNode) {
                        if (this.hoveredNode === n1 || this.hoveredNode === n2) {
                            alpha = 0.65;
                            strokeColor = 'rgba(56, 189, 248, 0.7)';
                        } else {
                            alpha = 0.02;
                        }
                    }

                    this.ctx.beginPath();
                    this.ctx.moveTo(n1.x, n1.y);
                    this.ctx.lineTo(n2.x, n2.y);
                    this.ctx.strokeStyle = strokeColor;
                    this.ctx.lineWidth = alpha > 0.5 ? 1.8 : 0.6;
                    this.ctx.stroke();
                }
            }
        }
        this.ctx.restore();
    }

    updateAndDrawParticles() {
        if (this.particles.length === 0) return;
        this.ctx.save();

        const activeParticles = [];
        let stepCompleted = true;

        for (const p of this.particles) {
            p.progress += p.speed;
            if (p.progress > 1) p.progress = 1;
            else stepCompleted = false;

            const curX = p.srcX + (p.dstX - p.srcX) * p.progress;
            const curY = p.srcY + (p.dstY - p.srcY) * p.progress;

            // Draw glowing particle
            this.ctx.beginPath();
            this.ctx.arc(curX, curY, p.size, 0, Math.PI * 2);
            this.ctx.fillStyle = p.color;
            this.ctx.shadowColor = p.color;
            this.ctx.shadowBlur = 8;
            this.ctx.fill();

            if (p.progress < 1) {
                activeParticles.push(p);
            }
        }
        this.ctx.restore();
        this.particles = activeParticles;

        if (stepCompleted && this.isAnimating) {
            if (this.currentStep === 1) {
                this.currentStep = 2;
                this.spawnParticlesForLayer(1, 2);
            } else if (this.currentStep === 2) {
                this.currentStep = 3;
                this.spawnParticlesForLayer(2, 3);
            } else {
                this.currentStep = 0;
                this.isAnimating = false;
                const event = new CustomEvent('nn-propagation-complete');
                window.dispatchEvent(event);
            }
        }
    }

    drawNeurons() {
        this.ctx.save();
        for (let l = 0; l < this.nodePositions.length; l++) {
            const layer = this.nodePositions[l];

            for (const node of layer) {
                const isHovered = this.hoveredNode === node;
                let activationVal = 0.5;

                if (this.activations) {
                    if (l === 0 && this.activations.inputs_scaled) {
                        const v = Math.abs(this.activations.inputs_scaled[node.index] || 0);
                        activationVal = Math.min(1.0, v / 3.0);
                    } else if (l === 1 && this.activations.layer1_a) {
                        const v = this.activations.layer1_a[node.index] || 0;
                        activationVal = Math.min(1.0, v / 5.0);
                    } else if (l === 2 && this.activations.layer2_a) {
                        const v = this.activations.layer2_a[node.index] || 0;
                        activationVal = Math.min(1.0, v / 4.0);
                    } else if (l === 3 && this.activations.layer3_probs) {
                        activationVal = this.activations.layer3_probs[node.index] || 0;
                    }
                }

                // Determine color palette
                let fillColor = '#1e293b';
                let strokeColor = '#475569';
                let glowColor = 'rgba(56, 189, 248, 0.4)';

                if (l === 0) {
                    strokeColor = '#38bdf8';
                    fillColor = `rgba(14, 165, 233, ${0.15 + activationVal * 0.7})`;
                } else if (l === 1 || l === 2) {
                    if (activationVal > 0.05) {
                        strokeColor = '#10b981';
                        fillColor = `rgba(16, 185, 129, ${0.15 + activationVal * 0.75})`;
                        glowColor = 'rgba(16, 185, 129, 0.6)';
                    } else {
                        strokeColor = '#334155';
                        fillColor = '#0f172a';
                    }
                } else if (l === 3) {
                    if (activationVal > 0.5) {
                        strokeColor = '#f59e0b';
                        fillColor = `rgba(245, 158, 11, ${0.25 + activationVal * 0.75})`;
                        glowColor = 'rgba(245, 158, 11, 0.8)';
                    } else {
                        strokeColor = '#64748b';
                        fillColor = '#1e293b';
                    }
                }

                // Outer halo if firing strongly or hovered
                if (activationVal > 0.4 || isHovered) {
                    this.ctx.beginPath();
                    this.ctx.arc(node.x, node.y, node.radius + (isHovered ? 6 : 4), 0, Math.PI * 2);
                    this.ctx.fillStyle = glowColor;
                    this.ctx.fill();
                }

                // Node circle
                this.ctx.beginPath();
                this.ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
                this.ctx.fillStyle = fillColor;
                this.ctx.fill();
                this.ctx.lineWidth = isHovered ? 2.5 : 1.5;
                this.ctx.strokeStyle = strokeColor;
                this.ctx.stroke();

                // Node label text inside or beside
                if (l === 0) {
                    const label = this.inputLabels[node.index] || `X${node.index}`;
                    this.ctx.fillStyle = '#f8fafc';
                    this.ctx.font = '600 10px Inter, sans-serif';
                    this.ctx.textAlign = 'center';
                    this.ctx.textBaseline = 'middle';
                    this.ctx.fillText(label, node.x, node.y);

                    // Measurement beside
                    const rawVal = window.CURRENT_INPUTS ? window.CURRENT_INPUTS[label] : '';
                    if (rawVal !== undefined && rawVal !== '') {
                        this.ctx.fillStyle = '#94a3b8';
                        this.ctx.font = '500 9px Inter, monospace';
                        this.ctx.textAlign = 'right';
                        this.ctx.fillText(String(rawVal), node.x - (this.width < 500 ? 14 : 18), node.y);
                    }
                }
            }
        }
        this.ctx.restore();
    }

    drawOutputLabels() {
        const outLayer = this.nodePositions[this.nodePositions.length - 1];
        if (!outLayer) return;
        this.ctx.save();

        for (const node of outLayer) {
            const cls = this.classes[node.index] || `Out ${node.index}`;
            const prob = this.activations && this.activations.layer3_probs ? this.activations.layer3_probs[node.index] : 0;
            const probPct = (prob * 100).toFixed(1);
            const isWinner = prob > 0.5;

            // Shorten class name for visual clarity in compact view
            let displayCls = cls.replace(" (A-B-C-G)", " (LLLG)")
                                .replace(" (A-B-C)", " (LLL)")
                                .replace(" (A-B-G)", " (LLG)")
                                .replace(" (B-C)", " (LL)")
                                .replace(" (A-G)", " (LG)")
                                .replace("3-Phase Symmetrical (LLL/LLLG)", "3-Ph Symm");

            // Class name
            this.ctx.fillStyle = isWinner ? '#f59e0b' : '#cbd5e1';
            this.ctx.font = isWinner ? '700 11px Inter, sans-serif' : '500 10px Inter, sans-serif';
            this.ctx.textAlign = 'left';
            this.ctx.textBaseline = 'middle';
            this.ctx.fillText(displayCls, node.x + 16, node.y - 4);

            // Probability mini bar
            const barWidth = Math.max(35, Math.min(55, this.width - (node.x + 16) - 45));
            const barHeight = 4;
            const barX = node.x + 16;
            const barY = node.y + 6;

            this.ctx.fillStyle = '#334155';
            this.ctx.fillRect(barX, barY, barWidth, barHeight);

            this.ctx.fillStyle = isWinner ? '#f59e0b' : '#38bdf8';
            this.ctx.fillRect(barX, barY, Math.max(0, barWidth * prob), barHeight);

            // Percentage text
            this.ctx.fillStyle = '#94a3b8';
            this.ctx.font = '500 9px Inter, monospace';
            this.ctx.fillText(`${probPct}%`, barX + barWidth + 5, barY + 3);
        }
        this.ctx.restore();
    }
}

window.NeuralNetworkVisualizer = NeuralNetworkVisualizer;

/**
 * Power System Visualizers:
 * 1. Three-Phase AC Oscilloscope (Va, Vb, Vc & Ia, Ib, Ic waveforms)
 * 2. Polar Phasor Diagram (120 deg balanced vs fault unbalance vectors)
 * 3. Transmission Line & Substation Circuit Breaker Arc Visualizer
 */

class ThreePhaseOscilloscope {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) return;
        this.ctx = this.canvas.getContext('2d');
        this.mode = 'current'; // 'current' or 'voltage'
        this.time = 0;
        this.freq = 50; // 50 Hz power grid
        this.isRunning = true;

        this.currentInputs = {
            Ia: 0, Ib: 0, Ic: 0,
            Va: 0.4, Vb: -0.2, Vc: -0.2
        };

        this.resize();
        window.addEventListener('resize', () => this.resize());
        this.startLoop();
    }

    setMode(mode) {
        this.mode = mode;
    }

    updateInputs(inputs) {
        if (!inputs) return;
        this.currentInputs = Object.assign(this.currentInputs, inputs);
    }

    resize() {
        if (!this.canvas) return;
        const rect = this.canvas.parentElement.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        this.width = rect.width;
        this.height = Math.max(220, Math.min(rect.width * 0.42, 280));
        this.canvas.width = this.width * dpr;
        this.canvas.height = this.height * dpr;
        this.canvas.style.width = `${this.width}px`;
        this.canvas.style.height = `${this.height}px`;
        this.ctx.scale(dpr, dpr);
    }

    startLoop() {
        const loop = () => {
            if (this.isRunning) {
                this.time += 0.04;
                this.render();
            }
            requestAnimationFrame(loop);
        };
        requestAnimationFrame(loop);
    }

    render() {
        if (!this.ctx || !this.width || !this.height) return;
        this.ctx.clearRect(0, 0, this.width, this.height);

        const midY = this.height / 2;
        const scaleY = this.height * 0.38;

        // Draw oscilloscope background grid
        this.drawGrid(midY);

        // Calculate normalized amplitudes based on inputs
        let ampA, ampB, ampC, ampN;
        if (this.mode === 'current') {
            // Normalize currents (-1000 to +1000 A)
            ampA = Math.max(-1.0, Math.min(1.0, this.currentInputs.Ia / 600));
            ampB = Math.max(-1.0, Math.min(1.0, this.currentInputs.Ib / 600));
            ampC = Math.max(-1.0, Math.min(1.0, this.currentInputs.Ic / 600));
            const ig = this.currentInputs.Ia + this.currentInputs.Ib + this.currentInputs.Ic;
            ampN = Math.max(-1.0, Math.min(1.0, ig / 600));
        } else {
            // Voltages (-0.8 to +0.8 p.u.)
            ampA = Math.max(-1.0, Math.min(1.0, this.currentInputs.Va / 0.5));
            ampB = Math.max(-1.0, Math.min(1.0, this.currentInputs.Vb / 0.5));
            ampC = Math.max(-1.0, Math.min(1.0, this.currentInputs.Vc / 0.5));
            const vg = this.currentInputs.Va + this.currentInputs.Vb + this.currentInputs.Vc;
            ampN = Math.max(-1.0, Math.min(1.0, vg / 0.5));
        }

        // Draw sine waves: Phase A (0 deg), Phase B (-120 deg), Phase C (+120 deg)
        this.drawSineWave(ampA, 0, '#ef4444', 2.5); // Phase A: Red
        this.drawSineWave(ampB, -2 * Math.PI / 3, '#eab308', 2.2); // Phase B: Yellow
        this.drawSineWave(ampC, 2 * Math.PI / 3, '#38bdf8', 2.2); // Phase C: Blue

        // Draw Neutral / Ground Residual Waveform (Dashed Green)
        if (Math.abs(ampN) > 0.02) {
            this.drawSineWave(ampN, 0, '#10b981', 1.8, true);
        }

        // Draw telemetry legends
        this.drawLegends();
    }

    drawGrid(midY) {
        this.ctx.save();
        this.ctx.strokeStyle = 'rgba(51, 65, 85, 0.45)';
        this.ctx.lineWidth = 1;

        // Horizontal division lines
        for (let y = 25; y < this.height; y += 35) {
            this.ctx.beginPath();
            this.ctx.moveTo(0, y);
            this.ctx.lineTo(this.width, y);
            this.ctx.stroke();
        }

        // Vertical division lines
        for (let x = 40; x < this.width; x += 50) {
            this.ctx.beginPath();
            this.ctx.moveTo(x, 0);
            this.ctx.lineTo(x, this.height);
            this.ctx.stroke();
        }

        // Central 0V/0A axis line
        this.ctx.strokeStyle = 'rgba(148, 163, 184, 0.7)';
        this.ctx.lineWidth = 1.5;
        this.ctx.setLineDash([5, 4]);
        this.ctx.beginPath();
        this.ctx.moveTo(0, midY);
        this.ctx.lineTo(this.width, midY);
        this.ctx.stroke();
        this.ctx.restore();
    }

    drawSineWave(amplitude, phaseOffset, color, lineWidth, isDashed = false) {
        this.ctx.save();
        this.ctx.beginPath();
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = lineWidth;
        this.ctx.shadowColor = color;
        this.ctx.shadowBlur = 10;
        if (isDashed) {
            this.ctx.setLineDash([7, 4]);
        }

        const midY = this.height / 2;
        const scaleY = (this.height / 2 - 32);
        const cycles = 2.5;

        for (let x = 0; x <= this.width; x += 2) {
            const angle = (x / this.width) * cycles * 2 * Math.PI - this.time + phaseOffset;
            const y = midY - Math.sin(angle) * (amplitude * scaleY);
            if (x === 0) this.ctx.moveTo(x, y);
            else this.ctx.lineTo(x, y);
        }
        this.ctx.stroke();
        this.ctx.restore();
    }

    drawLegends() {
        this.ctx.save();
        const prefix = this.mode === 'current' ? 'I' : 'V';
        const unit = this.mode === 'current' ? 'A' : 'p.u.';
        const ig = (this.currentInputs.Ia + this.currentInputs.Ib + this.currentInputs.Ic).toFixed(1);
        const badges = [
            { label: `${prefix}a: ${this.currentInputs[prefix + 'a']} ${unit}`, color: '#ef4444' },
            { label: `${prefix}b: ${this.currentInputs[prefix + 'b']} ${unit}`, color: '#eab308' },
            { label: `${prefix}c: ${this.currentInputs[prefix + 'c']} ${unit}`, color: '#38bdf8' }
        ];

        let startX = 14;
        for (const b of badges) {
            this.ctx.save();
            this.ctx.fillStyle = b.color;
            this.ctx.shadowColor = b.color;
            this.ctx.shadowBlur = 8;
            this.ctx.beginPath();
            this.ctx.arc(startX, 18, 5, 0, Math.PI * 2);
            this.ctx.fill();
            this.ctx.restore();

            this.ctx.fillStyle = '#f8fafc';
            this.ctx.font = '700 11px Inter, monospace';
            this.ctx.fillText(b.label, startX + 9, 21);
            startX += 115;
        }

        // Draw In residual if non-zero
        if (this.mode === 'current' && Math.abs(parseFloat(ig)) > 20) {
            this.ctx.save();
            this.ctx.fillStyle = '#10b981';
            this.ctx.shadowColor = '#10b981';
            this.ctx.shadowBlur = 8;
            this.ctx.beginPath();
            this.ctx.arc(startX, 18, 5, 0, Math.PI * 2);
            this.ctx.fill();
            this.ctx.restore();

            this.ctx.fillStyle = '#34d399';
            this.ctx.font = '700 11px Inter, monospace';
            this.ctx.fillText(`In(G): ${ig} A`, startX + 9, 21);
        }

        // Top right mode indicator
        this.ctx.fillStyle = '#94a3b8';
        this.ctx.font = '600 11px Inter, sans-serif';
        this.ctx.textAlign = 'right';
        this.ctx.fillText(`⚡ 50Hz 3-Phase ${this.mode.toUpperCase()} SCOPE`, this.width - 14, 20);

        this.ctx.restore();
    }
}


class PhasorDiagram {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) return;
        this.ctx = this.canvas.getContext('2d');
        this.currentInputs = { Ia: 0, Ib: 0, Ic: 0, Va: 0.4, Vb: -0.2, Vc: -0.2 };
        this.resize();
        window.addEventListener('resize', () => this.resize());
        this.render();
    }

    updateInputs(inputs) {
        if (!inputs) return;
        this.currentInputs = Object.assign(this.currentInputs, inputs);
        this.render();
    }

    resize() {
        if (!this.canvas) return;
        const rect = this.canvas.parentElement.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        this.size = Math.max(220, Math.min(rect.width, 280));
        this.canvas.width = this.size * dpr;
        this.canvas.height = this.size * dpr;
        this.canvas.style.width = `${this.size}px`;
        this.canvas.style.height = `${this.size}px`;
        this.ctx.scale(dpr, dpr);
        this.render();
    }

    render() {
        if (!this.ctx || !this.size) return;
        this.ctx.clearRect(0, 0, this.size, this.size);

        const cx = this.size / 2;
        const cy = this.size / 2;
        const maxR = this.size * 0.40;

        this.ctx.save();
        // Polar radar rings (0.25, 0.5, 0.75, 1.0 p.u.)
        this.ctx.strokeStyle = 'rgba(71, 85, 105, 0.35)';
        this.ctx.lineWidth = 1;
        for (const frac of [0.25, 0.5, 0.75, 1.0]) {
            this.ctx.beginPath();
            this.ctx.arc(cx, cy, maxR * frac, 0, Math.PI * 2);
            this.ctx.stroke();
        }

        // Axes (Crosshair)
        this.ctx.strokeStyle = 'rgba(100, 116, 139, 0.5)';
        this.ctx.beginPath();
        this.ctx.moveTo(cx - maxR - 8, cy);
        this.ctx.lineTo(cx + maxR + 8, cy);
        this.ctx.moveTo(cx, cy - maxR - 8);
        this.ctx.lineTo(cx, cy + maxR + 8);
        this.ctx.stroke();

        // Balanced 120-deg reference radial lines
        this.ctx.setLineDash([3, 4]);
        this.ctx.strokeStyle = 'rgba(148, 163, 184, 0.25)';
        for (const deg of [0, 120, 240]) {
            const rad = (deg * Math.PI) / 180;
            this.ctx.beginPath();
            this.ctx.moveTo(cx, cy);
            this.ctx.lineTo(cx + Math.cos(rad) * maxR, cy - Math.sin(rad) * maxR);
            this.ctx.stroke();
        }
        this.ctx.setLineDash([]);

        // Compute phasor magnitudes from currents (normalized to 650A base)
        const baseCurrent = 650;
        const ia = this.currentInputs.Ia || 0;
        const ib = this.currentInputs.Ib || 0;
        const ic = this.currentInputs.Ic || 0;

        const magA = Math.max(0.1, Math.min(1.0, Math.abs(ia) / baseCurrent));
        const magB = Math.max(0.1, Math.min(1.0, Math.abs(ib) / baseCurrent));
        const magC = Math.max(0.1, Math.min(1.0, Math.abs(ic) / baseCurrent));

        // Invert angle if instantaneous current is negative (180 deg phase flip)
        const signA = ia < 0 ? Math.PI : 0;
        const signB = ib < 0 ? Math.PI : 0;
        const signC = ic < 0 ? Math.PI : 0;

        const angA = 0 + signA;
        const angB = (-120 * Math.PI / 180) + signB;
        const angC = (120 * Math.PI / 180) + signC;

        // Draw 3 Phase Vectors with high-visibility neon glow
        this.drawVector(cx, cy, magA * maxR, angA, '#ef4444', `Ia (${Math.round(ia)}A)`);
        this.drawVector(cx, cy, magB * maxR, angB, '#eab308', `Ib (${Math.round(ib)}A)`);
        this.drawVector(cx, cy, magC * maxR, angC, '#38bdf8', `Ic (${Math.round(ic)}A)`);

        // Calculate Neutral/Ground Unbalance Vector In = Ia + Ib + Ic
        const vx_n = ia * Math.cos(signA) + ib * Math.cos(-2 * Math.PI / 3 + signB) + ic * Math.cos(2 * Math.PI / 3 + signC);
        const vy_n = ia * Math.sin(signA) + ib * Math.sin(-2 * Math.PI / 3 + signB) + ic * Math.sin(2 * Math.PI / 3 + signC);
        const in_mag = Math.sqrt(vx_n * vx_n + vy_n * vy_n);

        if (in_mag > 35) {
            const in_ang = Math.atan2(vy_n, vx_n);
            const in_len = Math.min(1.0, in_mag / baseCurrent) * maxR;
            this.drawVector(cx, cy, in_len, -in_ang, '#10b981', `In Ground Surge (${Math.round(in_mag)}A)`, true);
        }

        // Center origin point
        this.ctx.fillStyle = '#f8fafc';
        this.ctx.beginPath();
        this.ctx.arc(cx, cy, 3.5, 0, Math.PI * 2);
        this.ctx.fill();

        this.ctx.restore();

        // Update balance status label in DOM if present
        const statusEl = document.getElementById('phasor-balance-status');
        if (statusEl) {
            if (in_mag > 60) {
                statusEl.innerHTML = `<span class="text-rose-400 font-bold">⚠ Unbalanced Ground Surge</span>: In = ${Math.round(in_mag)} A`;
            } else if (in_mag > 25) {
                statusEl.innerHTML = `<span class="text-amber-400 font-semibold">⚡ Slight Line Asymmetry</span>: In = ${Math.round(in_mag)} A`;
            } else {
                statusEl.innerHTML = `<span class="text-emerald-400 font-semibold">✔ Symmetrical 3-Phase Balance</span> (In ≈ 0)`;
            }
        }
    }

    drawVector(cx, cy, length, angleRad, color, label, isDashed = false) {
        const vx = cx + Math.cos(angleRad) * length;
        const vy = cy - Math.sin(angleRad) * length;

        this.ctx.save();
        this.ctx.strokeStyle = color;
        this.ctx.fillStyle = color;
        this.ctx.lineWidth = isDashed ? 2.2 : 3.0;
        this.ctx.shadowColor = color;
        this.ctx.shadowBlur = 10;

        if (isDashed) {
            this.ctx.setLineDash([5, 4]);
        }

        // Line
        this.ctx.beginPath();
        this.ctx.moveTo(cx, cy);
        this.ctx.lineTo(vx, vy);
        this.ctx.stroke();

        // Arrow head
        const headlen = 9;
        const angle = Math.atan2(vy - cy, vx - cx);
        this.ctx.beginPath();
        this.ctx.moveTo(vx, vy);
        this.ctx.lineTo(vx - headlen * Math.cos(angle - Math.PI / 7), vy - headlen * Math.sin(angle - Math.PI / 7));
        this.ctx.lineTo(vx - headlen * Math.cos(angle + Math.PI / 7), vy - headlen * Math.sin(angle + Math.PI / 7));
        this.ctx.closePath();
        this.ctx.fill();

        // Label with background pill for contrast
        this.ctx.shadowBlur = 0;
        this.ctx.font = '700 10.5px Inter, monospace';
        const lx = cx + Math.cos(angleRad) * (length + 15);
        const ly = cy - Math.sin(angleRad) * (length + 15);
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';
        this.ctx.fillText(label, lx, ly);

        this.ctx.restore();
    }
}


class TransmissionGridVisualizer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.statusEl = document.getElementById('grid-breaker-status');
        this.arcEl = document.getElementById('grid-arc-effect');
        this.faultTextEl = document.getElementById('grid-fault-text');
    }

    updateFaultState(predictedFault) {
        if (!this.statusEl || !this.faultTextEl) return;
        const isFault = predictedFault && !predictedFault.includes("No Fault");

        if (!isFault) {
            this.statusEl.className = 'breaker-badge breaker-closed';
            this.statusEl.innerHTML = '<span class="status-dot green"></span> SUBSTATION CB: CLOSED (HEALTHY)';
            this.faultTextEl.innerHTML = '<span class="text-emerald font-semibold">Normal 3-Phase Transmission</span> - All conductors energized within safe capacity.';
            if (this.arcEl) this.arcEl.style.display = 'none';
        } else {
            this.statusEl.className = 'breaker-badge breaker-tripped';
            this.statusEl.innerHTML = '<span class="status-dot red animate-ping"></span> SUBSTATION CB: TRIPPED (ISOLATED)';
            this.faultTextEl.innerHTML = `<span class="text-crimson font-bold">FAULT DETECTED: ${predictedFault}</span> - Protective relay trip command sent in <span class="text-amber">11.8 ms</span>.`;
            if (this.arcEl) {
                this.arcEl.style.display = 'block';
                this.arcEl.className = 'arc-flash active';
            }
        }
    }
}

window.ThreePhaseOscilloscope = ThreePhaseOscilloscope;
window.PhasorDiagram = PhasorDiagram;
window.TransmissionGridVisualizer = TransmissionGridVisualizer;

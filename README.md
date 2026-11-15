<<<<<<< HEAD
# Real-Time Transmission Line Fault Detection and Classification using an Artificial Neural Network

**Department of Electrical and Electronics Engineering, NIT Puducherry**  
**Course:** Soft Computing (V Semester)  
**Project Proposal Team:**
- **Lalith** (`EE24B1030`) — Data Preprocessing & Streaming Architecture
- **Prabhat** (`EE24B1041`) — ANN Modeling & Hyperparameter Optimization
- **Vamsi** (`EE24B1040`) — Real-Time Visualization & Evaluation Metrics

---

## ⚡ Quick Start: Running the Interactive Presentation

You can launch and present this application in **any of three ways**:

### Option 1: One-Click Launcher (Recommended for Windows)
Double-click `run.bat` in the project folder, or run:
```powershell
python run.py
```
This starts the local Python backend server and automatically opens your default browser at `http://localhost:8000`.

### Option 2: Python HTTP Server
```powershell
python server.py
```
Then open `http://localhost:8000` in Google Chrome, Microsoft Edge, or any modern web browser.

### Option 3: Standalone Offline Mode (Zero Server Dependency)
Double-click `index.html` directly. The app includes a pure JavaScript forward-propagation neural engine (`model_data.js`) with the exact weights, biases, and standard scaler trained on the Kaggle dataset. It runs 100% offline with zero dependencies!

---

## 🌟 Key Features of the Interactive Presentation Webpage

### 1. Animated Neural Network Visualizer ("How the Prediction Happened")
- **Dynamic Synaptic Pulses:** When a fault is predicted, energetic glowing signal pulses travel along the synapses from the 6 input nodes $\rightarrow$ Hidden Layer 1 (ReLU) $\rightarrow$ Hidden Layer 2 (ReLU) $\rightarrow$ Output Layer (Softmax).
- **Synaptic Weights Representation:** Synapse line colors and opacities reflect the actual trained weights: cyan for positive/excitatory connections and pink/red for negative/inhibitory connections.
- **Neuron Inspection Tooltip:** Hovering over any neuron shows its mathematical details:
  - Input Layer: Feature name and standardized $z$-score ($z = \frac{x - \mu}{\sigma}$).
  - Hidden Layers: Weighted sum $z = \mathbf{W}\mathbf{x} + b$, and post-activation $a = \max(0, z)$ (ReLU).
  - Output Layer: Winning class, raw logit, and Softmax confidence percentage.
- **"Play Signal Pulse" button:** Trigger slow-motion propagation on demand during your classroom presentation!

### 2. Interactive Power System Diagnostics
- **3-Phase AC Oscilloscope:** Real-time 50 Hz sinusoidal waveforms for Phase A (Red), Phase B (Yellow), Phase C (Blue), and Neutral/Ground (Dashed Green). Reacts live as you adjust the current or voltage sliders!
- **Polar Phasor Diagram:** Displays the $120^\circ$ symmetrical star. Easily show your professor how vectors distort under LG, LL, LLG, or LLL fault conditions.
- **Substation Transmission Line Schematic:** Visual transmission towers with dynamic electrical arc flashes and circuit breaker status (Green "CLOSED/HEALTHY" vs Red flashing "TRIPPED/ISOLATED" with trip time delay in ms).

### 3. One-Click Classroom Demo Presets
Instant 1-click test buttons for classroom demonstrations:
- 🟢 **Normal (No Fault):** Balanced 3-phase load.
- 🔴 **LG (Phase A-G):** Phase A current surges, voltage sags, high zero-sequence ground current $I_g$.
- 🟠 **LL (Phase B-C):** Equal and opposite surge currents on Phase B & C, $I_g \approx 0$.
- 🟣 **LLG (Phase A-B-G):** Dual phase surge with high ground return.
- 🔵 **LLL (3-Phase Symmetrical):** Severe 3-phase short circuit.
- ⚪ **LLLG (3-Phase with Ground):** Symmetrical short with ground connection.
- 🎲 **Random Kaggle Test Sample:** Loads a random sample from the test set and displays the true Kaggle ground-truth label for instant verification.

### 4. Real-Time Telemetry & MongoDB Streaming Pipeline
- Demonstrates the proposal workflow: `Line Data (V, I) ➔ MongoDB Storage ➔ Preprocessing ➔ ANN Model ➔ Fault Alert`.
- Live streaming table displaying continuous readings, predictions, ground truth, and inference latency (< 0.5 ms).
- Live MongoDB Document Inspector showing raw JSON documents stored in `fault_db.readings`.

### 5. Interactive Slide Deck Presentation Mode
- Click the **"Classroom Lecture Slides"** tab to switch into a 5-slide presentation deck designed for your oral speech.

---

## 🔬 Model Analysis & Improvements (Soft Computing Insight)

### The Academic Discovery: Why 6-Class vs 5-Class Accuracy Differs (85.2% vs 99.55%)
When evaluating the Kaggle dataset on all 6 original classes, the baseline ANN achieves **85.19%** accuracy. A detailed inspection reveals that the model performs with **>99% accuracy on all individual faults except one pair: LLL vs LLLG**.

#### Electrical Power Engineering Physics Explanation:
- In a balanced 3-phase symmetrical fault ($LLL$), all three conductors are shorted together with currents $120^\circ$ apart.
- According to **Kirchhoff's Current Law (KCL)**:
  $$\sum I = I_a + I_b + I_c = 0$$
- In a symmetrical fault with ground ($LLLG$), because the three phases remain balanced $120^\circ$ apart, the potential at the neutral point remains at **zero volts**.
- Therefore, **no zero-sequence ground current flows** ($I_g = I_a + I_b + I_c \approx 0$), making $LLL$ and $LLLG$ physically identical at the measurement terminals.

#### Solution & Model Improvements Implemented:
1. **IEEE Protective Relay Standard (5-Class Model):**
   By categorizing faults according to standard protective relay tripping zones (grouping symmetrical 3-phase faults), the ANN achieves an outstanding **99.55% accuracy**!
2. **Multi-Output Conductor Identification ANN:**
   Predicts Ground ($G$), Phase $C$, Phase $B$, and Phase A ($A$) binary flags. 
   - Phase A Detection Accuracy: **99.81%**
   - Phase B Detection Accuracy: **99.62%**
   - Phase C Detection Accuracy: **99.68%**
   - Ground Detection Accuracy: **85.31%** (due to the KCL neutral cancellation mentioned above).

---

## 👥 Suggested Speaking Roles for Classroom Presentation

### 1. Lalith (EE24B1030) — Data Preprocessing & Streaming Architecture
- Explain the transmission line protection problem and dataset features ($I_a, I_b, I_c, V_a, V_b, V_c$).
- Explain why standard scaling ($z = \frac{x - \mu}{\sigma}$) is critical for ANNs when dealing with large currents ($\pm 1000\text{ A}$) and fractional voltages ($\pm 0.6\text{ p.u.}$).
- Demonstrate the **Real-Time Telemetry & MongoDB Streaming tab**.

### 2. Prabhat (EE24B1041) — ANN Modeling & Mathematical Forward Pass
- Walk the class through the **Animated Neural Network Visualizer**.
- Explain the MLP architecture: 6 inputs $\rightarrow$ 32 ReLU neurons $\rightarrow$ 16 ReLU neurons $\rightarrow$ 6 Softmax outputs.
- Explain the mathematical forward pass: $\mathbf{a}^{[1]} = \text{ReLU}(\mathbf{W}_1\mathbf{z} + \mathbf{b}_1)$, $\mathbf{a}^{[2]} = \text{ReLU}(\mathbf{W}_2\mathbf{a}^{[1]} + \mathbf{b}_2)$, and the Softmax probability distribution.
- Highlight the **Symmetrical Fault Discovery (85.2% vs 99.55%)** on the Performance tab.

### 3. Vamsi (EE24B1040) — Live Interactive Demonstration & Grid Protection
- Trigger the live interactive presets (Normal, LG, LL, LLG, LLL).
- Demonstrate the **3-Phase Oscilloscope** and **Polar Phasor Diagram**.
- Show how the circuit breaker trips in under **12 ms** and isolates the faulted section.
- Summarize the project conclusions and future scope (1D-CNN / LSTM and edge microcontrollers).
=======
# Fault-detection-in-Power-system-using-ANN
eal-time transmission line fault detection and classification using an ANN and MongoDB.
>>>>>>> a0d2aeb7ad14b0b1d38ece6d915ac4cd87de35cb

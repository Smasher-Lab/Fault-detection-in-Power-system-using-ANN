"""
Python HTTP Server for Real-Time Transmission Line Fault Detection & Classification
Department of Electrical and Electronics Engineering, NIT Puducherry
Course: Soft Computing
Team: Lalith (EE24B1030), Prabhat (EE24B1041), Vamsi (EE24B1040)

Features:
- Serves interactive presentation dashboard
- REST API for model predictions, layer activations, and explainability
- Real-time MongoDB streaming with robust in-memory fallback
- Zero extra dependencies (uses built-in http.server, works on Python 3.10+)
"""

import os
import sys
import json
import time
import math
from datetime import datetime, timezone
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.parse

import numpy as np
import pandas as pd
import joblib

from data import FEATURES, load_data

PORT = int(os.getenv("PORT", 8000))
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")

# Global variables for models and data
MODELS = {}
SCALER = None
DATA_DF = None
STREAM_INDEX = 0
TELEMETRY_LOG = []
MONGO_CLIENT = None
MONGO_AVAILABLE = False

def get_mongo_collection():
    """
    Dynamically connects or verifies MongoDB connection on demand.
    Returns the collection 'fault_db.readings' if reachable, else None.
    """
    global MONGO_CLIENT, MONGO_AVAILABLE
    try:
        from pymongo import MongoClient
        if MONGO_CLIENT is None:
            MONGO_CLIENT = MongoClient(MONGO_URI, serverSelectionTimeoutMS=1500)
        MONGO_CLIENT.server_info()  # ping server
        MONGO_AVAILABLE = True
        return MONGO_CLIENT["fault_db"]["readings"]
    except Exception as e:
        MONGO_AVAILABLE = False
        return None

def init_system():
    global MODELS, SCALER, DATA_DF, MONGO_CLIENT, MONGO_AVAILABLE
    print("[INIT] Loading trained models and dataset...")
    try:
        ann_base, scaler = joblib.load("fault_ann.joblib")
        MODELS["baseline"] = ann_base
        SCALER = scaler
        print("  - Loaded Baseline ANN")
    except Exception as e:
        print(f"  ! Warning loading baseline ANN: {e}")

    try:
        ann_enh, _ = joblib.load("fault_ann_enhanced.joblib")
        MODELS["enhanced"] = ann_enh
        print("  - Loaded Enhanced Deep ANN")
    except Exception as e:
        print(f"  ! Warning loading enhanced ANN: {e}")

    try:
        ann_5c, _ = joblib.load("fault_ann_5class.joblib")
        MODELS["protective_standard"] = ann_5c
        print("  - Loaded 5-Class Protective ANN")
    except Exception as e:
        print(f"  ! Warning loading 5-class ANN: {e}")

    try:
        # Shuffle dataset so simulated streaming yields a diverse mix of random faults
        DATA_DF = load_data().sample(frac=1.0, random_state=None).reset_index(drop=True)
        print(f"  - Loaded & Shuffled Kaggle Dataset ({len(DATA_DF)} rows)")
    except Exception as e:
        print(f"  ! Warning loading dataset: {e}")

    # Check initial MongoDB connectivity
    col = get_mongo_collection()
    if col is not None:
        count = col.count_documents({})
        print(f"  - MongoDB connected successfully at {MONGO_URI} (fault_db.readings: {count} docs)")
    else:
        print(f"  - MongoDB not reachable on {MONGO_URI}. In-memory fallback active.")


def forward_pass_explain(x_raw, model_key="baseline"):
    """
    Computes manual forward propagation through the chosen MLP and returns
    layer-by-layer activations, z-scores, and physical power engineering analysis.
    """
    model = MODELS.get(model_key, MODELS.get("baseline"))
    if model is None or SCALER is None:
        return None

    # 1. Feature scaling
    x_scaled = (np.array(x_raw) - SCALER.mean_) / SCALER.scale_

    # 2. Layer-by-layer propagation
    W1, W2, W3 = model.coefs_
    b1, b2, b3 = model.intercepts_

    # Layer 1
    z1 = np.dot(x_scaled, W1) + b1
    a1 = np.maximum(0, z1)  # ReLU

    # Layer 2
    z2 = np.dot(a1, W2) + b2
    a2 = np.maximum(0, z2)  # ReLU

    # Layer 3 (Logits -> Softmax)
    z3 = np.dot(a2, W3) + b3
    exp_z = np.exp(z3 - np.max(z3))
    probs = exp_z / np.sum(exp_z)

    pred_idx = int(np.argmax(probs))
    pred_class = model.classes_[pred_idx]
    confidence = float(probs[pred_idx])

    # Power system physics calculations
    ia, ib, ic, va, vb, vc = x_raw
    ig = ia + ib + ic  # Ground / Neutral residual current
    vg = va + vb + vc  # Zero-sequence residual voltage
    i_diff_ab = abs(ia - ib)
    i_diff_bc = abs(ib - ic)
    i_diff_ca = abs(ic - ia)

    # Physical reason deduction
    explanation = []
    if "No Fault" in pred_class:
        explanation.append("Balanced 3-Phase currents and voltages operating within continuous rated capacity.")
        explanation.append(f"Ground residual current Ig = {ig:.2f} A is negligible.")
    elif "LG" in pred_class:
        explanation.append(f"High ground fault return current detected: Ig = {ig:.2f} A.")
        explanation.append(f"Phase A current surges to {ia:.1f} A with corresponding phase voltage sag (Va = {va:.3f} p.u.).")
        explanation.append("Classic Single Line-to-Ground (A-G) signature.")
    elif "LLG" in pred_class:
        explanation.append(f"Simultaneous Phase A and B current surges with significant ground return: Ig = {ig:.2f} A.")
        explanation.append(f"Dual phase-to-ground short circuit (A-B-G).")
    elif "LL" in pred_class:
        explanation.append(f"Equal and opposite surge currents on Phase B ({ib:.1f} A) and Phase C ({ic:.1f} A).")
        explanation.append(f"Ground return Ig = {ig:.2f} A is nearly zero, confirming isolated phase-to-phase contact without ground.")
    elif "LLL" in pred_class:
        explanation.append("Severe symmetrical 3-phase short circuit across all conductors (A-B-C).")
        explanation.append(f"Phase voltages severely depressed (Va={va:.3f}, Vb={vb:.3f}, Vc={vc:.3f} p.u.).")
        explanation.append(f"Kirchhoff's Current Law holds: Ia + Ib + Ic = {ig:.2f} A ≈ 0 A due to 120° physical symmetry.")

    return {
        "model_used": model_key,
        "predicted_fault": str(pred_class),
        "confidence": round(confidence * 100, 2),
        "probabilities": {str(c): round(float(p) * 100, 2) for c, p in zip(model.classes_, probs)},
        "activations": {
            "inputs_scaled": [round(float(v), 4) for v in x_scaled],
            "layer1_z": [round(float(v), 4) for v in z1],
            "layer1_a": [round(float(v), 4) for v in a1],
            "layer2_z": [round(float(v), 4) for v in z2],
            "layer2_a": [round(float(v), 4) for v in a2],
            "layer3_z": [round(float(v), 4) for v in z3],
            "layer3_probs": [round(float(v), 4) for v in probs],
        },
        "power_physics": {
            "ground_current_ig": round(float(ig), 2),
            "zero_sequence_voltage_vg": round(float(vg), 4),
            "current_diffs": {
                "ab": round(float(i_diff_ab), 2),
                "bc": round(float(i_diff_bc), 2),
                "ca": round(float(i_diff_ca), 2),
            },
            "electrical_explanation": explanation,
        }
    }


class FaultPresentationHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        url_path = urllib.parse.urlparse(self.path).path

        if url_path == "/api/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            col = get_mongo_collection()
            doc_count = col.count_documents({}) if col is not None else len(TELEMETRY_LOG)
            status_obj = {
                "server": "online",
                "department": "EEE, NIT Puducherry",
                "mongodb_connected": col is not None,
                "database": "fault_db",
                "collection": "readings",
                "document_count": doc_count,
                "dataset_loaded": DATA_DF is not None,
                "models_available": list(MODELS.keys()),
            }
            self.wfile.write(json.dumps(status_obj).encode("utf-8"))
            return

        elif url_path == "/api/stream/next":
            global STREAM_INDEX, TELEMETRY_LOG
            if DATA_DF is None or len(DATA_DF) == 0:
                self.send_error(500, "Dataset not loaded")
                return

            row = DATA_DF.iloc[STREAM_INDEX % len(DATA_DF)]
            STREAM_INDEX += 1

            raw_vals = [float(row[f]) for f in FEATURES]
            t_start = time.perf_counter()
            pred_info = forward_pass_explain(raw_vals, "baseline")
            inference_ms = round((time.perf_counter() - t_start) * 1000, 3)

            now_str = datetime.now(timezone.utc).isoformat()
            col = get_mongo_collection()
            record = {
                "timestamp": now_str,
                "Ia": round(raw_vals[0], 2),
                "Ib": round(raw_vals[1], 2),
                "Ic": round(raw_vals[2], 2),
                "Va": round(raw_vals[3], 4),
                "Vb": round(raw_vals[4], 4),
                "Vc": round(raw_vals[5], 4),
                "actual_fault": row["fault"],
                "predicted_fault": pred_info["predicted_fault"] if pred_info else "Unknown",
                "confidence": pred_info["confidence"] if pred_info else 0,
                "inference_ms": inference_ms,
                "status": "MATCH" if (pred_info and pred_info["predicted_fault"] == row["fault"]) else "MISMATCH",
                "db_storage": "MongoDB [fault_db.readings]" if col is not None else "Memory Store (Simulated MongoDB)",
            }

            # If MongoDB is reachable, insert document
            doc_id = None
            if col is not None:
                try:
                    res = col.insert_one(record.copy())
                    doc_id = str(res.inserted_id)
                    record["_id"] = doc_id
                except Exception as e:
                    print(f"MongoDB write warning: {e}")

            TELEMETRY_LOG.append(record)
            if len(TELEMETRY_LOG) > 50:
                TELEMETRY_LOG.pop(0)

            total_docs = col.count_documents({}) if col is not None else len(TELEMETRY_LOG)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            response_data = {
                "record": record,
                "document_count": total_docs,
                "mongodb_connected": col is not None,
                "explanation": pred_info["power_physics"]["electrical_explanation"] if pred_info else [],
                "layer_activations": pred_info["activations"] if pred_info else {},
            }
            self.wfile.write(json.dumps(response_data).encode("utf-8"))
            return

        elif url_path in ["/api/stream/history", "/api/mongo/readings"]:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            col = get_mongo_collection()
            if col is not None:
                try:
                    docs = list(col.find().sort([("_id", -1)]).limit(30))
                    for d in docs:
                        d["_id"] = str(d["_id"])
                    self.wfile.write(json.dumps(docs).encode("utf-8"))
                    return
                except Exception as e:
                    print(f"Error reading MongoDB history: {e}")
            self.wfile.write(json.dumps(TELEMETRY_LOG[-30:]).encode("utf-8"))
            return

        # Serve static files
        super().do_GET()

    def do_POST(self):
        url_path = urllib.parse.urlparse(self.path).path

        if url_path == "/api/predict":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                payload = json.loads(body.decode("utf-8"))
                ia = float(payload.get("Ia", 0.0))
                ib = float(payload.get("Ib", 0.0))
                ic = float(payload.get("Ic", 0.0))
                va = float(payload.get("Va", 0.0))
                vb = float(payload.get("Vb", 0.0))
                vc = float(payload.get("Vc", 0.0))
                model_type = payload.get("model_type", "baseline")
                save_to_db = payload.get("save_to_db", True)

                t_start = time.perf_counter()
                result = forward_pass_explain([ia, ib, ic, va, vb, vc], model_type)
                latency_ms = round((time.perf_counter() - t_start) * 1000, 3)

                if result:
                    result["latency_ms"] = latency_ms

                # Optionally log event to MongoDB
                col = get_mongo_collection()
                doc_count = 0
                if save_to_db and result:
                    event_doc = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "Ia": round(ia, 2),
                        "Ib": round(ib, 2),
                        "Ic": round(ic, 2),
                        "Va": round(va, 4),
                        "Vb": round(vb, 4),
                        "Vc": round(vc, 4),
                        "actual_fault": payload.get("actual_fault", result["predicted_fault"]),
                        "predicted_fault": result["predicted_fault"],
                        "confidence": result["confidence"],
                        "model_used": model_type,
                        "source": payload.get("source", "Dashboard Prediction"),
                    }
                    if col is not None:
                        try:
                            res = col.insert_one(event_doc)
                            result["mongodb_id"] = str(res.inserted_id)
                            doc_count = col.count_documents({})
                        except Exception as e:
                            print(f"Failed to save predict event to MongoDB: {e}")
                    else:
                        TELEMETRY_LOG.append(event_doc)
                        doc_count = len(TELEMETRY_LOG)
                
                result["mongodb_connected"] = col is not None
                result["document_count"] = doc_count

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        elif url_path == "/api/mongo/save":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                payload = json.loads(body.decode("utf-8"))
                col = get_mongo_collection()
                doc = {
                    "timestamp": payload.get("timestamp", datetime.now(timezone.utc).isoformat()),
                    "Ia": round(float(payload.get("Ia", 0.0)), 2),
                    "Ib": round(float(payload.get("Ib", 0.0)), 2),
                    "Ic": round(float(payload.get("Ic", 0.0)), 2),
                    "Va": round(float(payload.get("Va", 0.0)), 4),
                    "Vb": round(float(payload.get("Vb", 0.0)), 4),
                    "Vc": round(float(payload.get("Vc", 0.0)), 4),
                    "actual_fault": str(payload.get("actual_fault", "Manual")),
                    "predicted_fault": str(payload.get("predicted_fault", "Unknown")),
                    "confidence": float(payload.get("confidence", 0.0)),
                    "model_used": str(payload.get("model_used", "baseline")),
                    "source": str(payload.get("source", "Manual Save")),
                }
                doc_id = None
                if col is not None:
                    res = col.insert_one(doc.copy())
                    doc_id = str(res.inserted_id)
                    doc["_id"] = doc_id
                    count = col.count_documents({})
                else:
                    doc["_id"] = len(TELEMETRY_LOG) + 1
                    TELEMETRY_LOG.append(doc)
                    count = len(TELEMETRY_LOG)

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "success": True,
                    "id": doc_id,
                    "document": doc,
                    "document_count": count,
                    "mongodb_connected": col is not None,
                    "message": "Saved to MongoDB (fault_db.readings)" if col is not None else "Saved to in-memory store"
                }).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        elif url_path == "/api/mongo/clear":
            try:
                col = get_mongo_collection()
                deleted_count = 0
                if col is not None:
                    res = col.delete_many({})
                    deleted_count = res.deleted_count
                TELEMETRY_LOG.clear()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "success": True,
                    "deleted_count": deleted_count,
                    "message": "MongoDB collection cleared successfully"
                }).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        self.send_error(404, "Endpoint not found")


def run_server():
    init_system()
    server_address = ("", PORT)
    httpd = HTTPServer(server_address, FaultPresentationHandler)
    print(f"\n========================================================")
    print(f"  NIT Puducherry - EEE Soft Computing Presentation App")
    print(f"  Team: Lalith (EE24B1030), Prabhat (EE24B1041), Vamsi (EE24B1040)")
    print(f"  Serving at: http://localhost:{PORT}")
    print(f"========================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        httpd.server_close()

if __name__ == "__main__":
    run_server()

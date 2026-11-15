"""Real-time fault detection demo: MongoDB + trained ANN.

Run in two terminals:
    python stream.py simulate     # pushes live-looking readings into MongoDB
    python stream.py predict      # picks them up, predicts fault type, writes it back
"""
import os, sys, time, random
from datetime import datetime, timezone

import joblib
import pandas as pd
from pymongo import MongoClient

from data import FEATURES, load_data

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
_FALLBACK_FILE = "simulated_mongo_readings.json"

def get_collection():
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=1500)
        client.server_info()
        return client["fault_db"]["readings"]
    except Exception:
        return None

def simulate(n=50, delay=1.0):
    """Replay random dataset rows as timestamped 'live' readings."""
    col = get_collection()
    using_mongo = col is not None
    if using_mongo:
        print("[MongoDB] Connected to MongoDB (fault_db.readings). Streaming live records...")
    else:
        print(f"[Simulated DB] MongoDB not reachable on {MONGO_URI}. Streaming to local buffer ({_FALLBACK_FILE})...")

    df = load_data().sample(n=n, random_state=None)
    for _, row in df.iterrows():
        doc = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **{f: float(row[f]) for f in FEATURES},
            "actual_fault": row["fault"],
            "predicted_fault": None,
        }
        if using_mongo:
            col.insert_one(doc)
        else:
            # Append to fallback JSON
            import json
            existing = []
            if os.path.exists(_FALLBACK_FILE):
                try:
                    with open(_FALLBACK_FILE, "r") as f:
                        existing = json.load(f)
                except Exception:
                    existing = []
            doc_id = len(existing) + 1
            doc["_id"] = doc_id
            existing.append(doc)
            with open(_FALLBACK_FILE, "w") as f:
                json.dump(existing, f, indent=2)

        print(f"[sim] reading streamed -> (actual: {row['fault']:<16}) Ia={row['Ia']:>7.1f} A, Va={row['Va']:>6.3f} p.u.")
        time.sleep(delay)


def predict_loop(poll=0.5, max_idle=None):
    """Find unpredicted readings, run the ANN, store the result."""
    ann, scaler = joblib.load("fault_ann.joblib")
    col = get_collection()
    using_mongo = col is not None
    if using_mongo:
        print("[MongoDB] Connected. Listening for unpredicted readings...")
    else:
        print(f"[Simulated DB] Listening for unpredicted readings in {_FALLBACK_FILE}...")

    idle_since = time.time()
    import json
    while True:
        batch = []
        if using_mongo:
            batch = list(col.find({"predicted_fault": None}).sort("timestamp", 1))
        else:
            if os.path.exists(_FALLBACK_FILE):
                try:
                    with open(_FALLBACK_FILE, "r") as f:
                        all_docs = json.load(f)
                    batch = [d for d in all_docs if d.get("predicted_fault") is None]
                except Exception:
                    batch = []

        for doc in batch:
            x = pd.DataFrame([[doc[f] for f in FEATURES]], columns=FEATURES)
            pred = ann.predict(scaler.transform(x))[0]

            if using_mongo:
                col.update_one({"_id": doc["_id"]}, {"$set": {
                    "predicted_fault": pred,
                    "predicted_at": datetime.now(timezone.utc),
                }})
            else:
                with open(_FALLBACK_FILE, "r") as f:
                    all_docs = json.load(f)
                for d in all_docs:
                    if d.get("_id") == doc["_id"]:
                        d["predicted_fault"] = pred
                        d["predicted_at"] = datetime.now(timezone.utc).isoformat()
                with open(_FALLBACK_FILE, "w") as f:
                    json.dump(all_docs, f, indent=2)

            flag = "OK  " if pred == doc["actual_fault"] else "MISS"
            alert = "" if pred == "No Fault" else "  << [CIRCUIT BREAKER TRIP ALERT]"
            print(f"[{flag}] predicted: {pred:<16} actual: {doc['actual_fault']:<16}{alert}")
            idle_since = time.time()
        if max_idle and time.time() - idle_since > max_idle:
            break
        time.sleep(poll)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "simulate":
        simulate()
    elif mode == "predict":
        predict_loop()
    else:
        print("Usage: python stream.py simulate | predict")
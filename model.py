import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from data import load_data, split_and_scale

def train():
    df = load_data()
    X_tr, X_te, y_tr, y_te, scaler = split_and_scale(df)
    ann = MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu",
                        max_iter=500, random_state=42)
    ann.fit(X_tr, y_tr)
    pred = ann.predict(X_te)
    print("Accuracy:", round(accuracy_score(y_te, pred), 4))
    print(classification_report(y_te, pred))
    print(confusion_matrix(y_te, pred, labels=ann.classes_))
    joblib.dump((ann, scaler), "fault_ann.joblib")

if __name__ == "__main__":
    train()

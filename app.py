"""Prediction API: send traffic features as JSON, get a threat verdict back.

Start:  python app.py        (http://127.0.0.1:5000)
"""
from flask import Flask, jsonify, request
from src.detector import ThreatDetector

app = Flask(__name__)
detector = ThreatDetector()


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": "random_forest + isolation_forest"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not data:
        return jsonify({"error": "Send a JSON object, e.g. {\"failed_logins\": 12}"}), 400
    try:
        return jsonify(detector.predict_one(data))
    except ValueError as err:
        return jsonify({"error": str(err)}), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
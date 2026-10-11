"""Send sample requests to the running API (start it first with: python app.py)."""
import json
import urllib.request

URL = "http://127.0.0.1:5000/predict"
SAMPLES = {
    "Normal browsing": {"packet_size": 480, "request_frequency": 4, "failed_logins": 0},
    "Brute-force login attempts": {"failed_logins": 15, "dst_port": 22, "request_frequency": 12},
    "Port scan": {"unique_ports": 45, "packet_size": 65, "syn_ratio": 0.7, "request_frequency": 60},
    "Request flood": {"request_frequency": 320, "syn_ratio": 0.8, "packet_size": 90},
    "Example from the project brief": {"packet_size": 1500, "failed_logins": 5, "request_frequency": 250},
}

for name, payload in SAMPLES.items():
    request = urllib.request.Request(URL, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        result = json.loads(response.read())
    print(f"\n{name}\n  sent    : {payload}\n  verdict : {result['attack_type']} "
          f"(score {result['attack_score']}, severity {result['severity']})")
    if result["reason"]:
        print(f"  reason  : {result['reason']}")
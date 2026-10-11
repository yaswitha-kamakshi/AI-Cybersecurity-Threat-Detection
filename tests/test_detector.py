import pytest
from src.detector import ThreatDetector


@pytest.fixture(scope="module")
def detector():
    return ThreatDetector()


def test_normal_traffic_is_not_flagged(detector):
    result = detector.predict_one({"packet_size": 480, "request_frequency": 4, "failed_logins": 0})
    assert result["threat_detected"] is False


def test_brute_force_is_detected(detector):
    result = detector.predict_one({"failed_logins": 15, "dst_port": 22})
    assert result["attack_type"] == "BruteForce"


def test_port_scan_is_detected(detector):
    result = detector.predict_one({"unique_ports": 45, "packet_size": 65, "syn_ratio": 0.7, "request_frequency": 60})
    assert result["attack_type"] == "PortScan"


def test_request_flood_is_detected(detector):
    result = detector.predict_one({"request_frequency": 320, "syn_ratio": 0.8, "packet_size": 90})
    assert result["attack_type"] == "DoS"


def test_bad_input_is_rejected(detector):
    with pytest.raises(ValueError):
        detector.predict_one({"failed_logins": "lots"})
"""Simulated network traffic: normal users plus DoS, brute-force and port-scan attacks.

Each row is one traffic "window" from one source IP (like a row of a NetFlow / IDS log).
The classes deliberately overlap (busy normal users, stealthy attackers) so the
detection problem is realistic and the model does not score a suspicious 100%.
"""
import os
import numpy as np
import pandas as pd

CLASSES = ["Normal", "DoS", "BruteForce", "PortScan"]
PROTOCOLS = ["tcp", "udp", "icmp"]
COMMON_PORTS = [80, 443, 53, 22, 25, 3306, 8080]


def _ips(rng, n, pool_size):
    """Random IPv4 strings drawn from a pool (small pool = repeat offenders)."""
    pool = [f"{rng.integers(11, 220)}.{rng.integers(0, 255)}.{rng.integers(0, 255)}.{rng.integers(1, 255)}"
            for _ in range(pool_size)]
    return rng.choice(pool, n)


def _normal_like(rng, n):
    """Feature draws that look like ordinary traffic (used by stealthy attackers to blend in)."""
    return {
        "packet_size": np.clip(rng.lognormal(6.0, 0.7, n), 40, 1500),
        "syn_ratio": rng.beta(1, 20, n),
        "src_bytes": rng.lognormal(6.5, 1.2, n),
        "dst_bytes": rng.lognormal(7.0, 1.3, n),
        "duration": rng.exponential(2.0, n),
    }


def _normal(rng, n):
    freq = np.where(rng.random(n) < 0.10,
                    rng.gamma(4, 12, n),            # busy users / bulk downloads
                    rng.gamma(2, 2.5, n))           # typical users
    failed = rng.poisson(0.15, n)
    typo = rng.random(n) < 0.08                     # forgot-password moments
    failed = np.where(typo, rng.integers(2, 6, n), failed)
    ports = 1 + rng.poisson(0.5, n)
    scanner_like = rng.random(n) < 0.05             # benign tools touching many ports
    ports = np.where(scanner_like, rng.integers(5, 14, n), ports)
    syn = np.where(rng.random(n) < 0.04, rng.beta(4, 6, n), rng.beta(1, 20, n))   # flaky connections
    base = _normal_like(rng, n)
    base["syn_ratio"] = syn
    return pd.DataFrame({
        **base,
        "request_frequency": freq,
        "failed_logins": failed,
        "unique_ports": ports,
        "protocol": rng.choice(PROTOCOLS, n, p=[0.75, 0.20, 0.05]),
        "dst_port": rng.choice(COMMON_PORTS, n, p=[0.3, 0.35, 0.1, 0.05, 0.05, 0.05, 0.1]),
        "src_ip": _ips(rng, n, 400),
    })


def _dos(rng, n):
    stealth = rng.random(n) < 0.40                  # low-rate DoS that hides among busy users
    freq = np.where(stealth, rng.uniform(15, 90, n), rng.lognormal(5.7, 0.6, n))
    flood_syn = rng.random(n) < 0.6                 # SYN flood vs plain request flood
    nl = _normal_like(rng, n)
    loud = {
        "packet_size": np.clip(np.where(flood_syn, rng.normal(90, 30, n), rng.normal(700, 300, n)), 40, 1500),
        "syn_ratio": np.where(flood_syn, rng.beta(8, 2, n), rng.beta(2, 8, n)),
        "src_bytes": rng.lognormal(5.5, 0.8, n),
        "dst_bytes": rng.lognormal(3.5, 1.0, n),
        "duration": rng.exponential(0.5, n),
    }
    return pd.DataFrame({
        **{k: np.where(stealth, nl[k], loud[k]) for k in loud},
        "request_frequency": freq,
        "failed_logins": rng.poisson(0.1, n),
        "unique_ports": 1 + rng.poisson(0.2, n),
        "protocol": rng.choice(PROTOCOLS, n, p=[0.6, 0.3, 0.1]),
        "dst_port": rng.choice([80, 443, 8080], n),
        "src_ip": _ips(rng, n, 25),
    })


def _brute(rng, n):
    stealth = rng.random(n) < 0.45                  # slow brute force, close to a forgetful user
    failed = np.where(stealth, rng.integers(3, 7, n), 6 + rng.poisson(10, n))
    nl = _normal_like(rng, n)
    loud = {
        "packet_size": np.clip(rng.normal(170, 60, n), 40, 1500),
        "syn_ratio": rng.beta(1.5, 12, n),
        "src_bytes": rng.lognormal(5.0, 0.6, n),
        "dst_bytes": rng.lognormal(5.0, 0.7, n),
        "duration": rng.exponential(1.0, n),
    }
    return pd.DataFrame({
        **{k: np.where(stealth, nl[k], loud[k]) for k in loud},
        "request_frequency": np.where(stealth, rng.uniform(1, 15, n), rng.uniform(3, 30, n)),
        "failed_logins": failed,
        "unique_ports": 1 + rng.poisson(0.1, n),
        "protocol": "tcp",
        "dst_port": rng.choice([22, 3389, 21], n, p=[0.6, 0.3, 0.1]),
        "src_ip": _ips(rng, n, 20),
    })


def _scan(rng, n):
    stealth = rng.random(n) < 0.45                  # slow scan touching few ports
    ports = np.where(stealth, rng.integers(6, 15, n), 20 + rng.poisson(30, n))
    nl = _normal_like(rng, n)
    loud = {
        "packet_size": np.clip(rng.normal(65, 15, n), 40, 1500),
        "syn_ratio": rng.beta(7, 3, n),
        "src_bytes": rng.lognormal(4.2, 0.5, n),
        "dst_bytes": rng.lognormal(3.0, 1.0, n),
        "duration": rng.exponential(0.3, n),
    }
    return pd.DataFrame({
        **{k: np.where(stealth, nl[k], loud[k]) for k in loud},
        "request_frequency": np.where(stealth, rng.uniform(3, 40, n), rng.uniform(10, 100, n)),
        "failed_logins": np.zeros(n, dtype=int),
        "unique_ports": ports,
        "protocol": rng.choice(PROTOCOLS, n, p=[0.85, 0.1, 0.05]),
        "dst_port": rng.choice(COMMON_PORTS + [21, 23, 445, 3389], n),
        "src_ip": _ips(rng, n, 15),
    })


def generate_traffic(n=20000, seed=42):
    """Return a shuffled DataFrame of simulated traffic with a `label` column."""
    rng = np.random.default_rng(seed)
    share = {"Normal": 0.80, "DoS": 0.08, "BruteForce": 0.07, "PortScan": 0.05}
    builders = {"Normal": _normal, "DoS": _dos, "BruteForce": _brute, "PortScan": _scan}
    parts = []
    for name in CLASSES:
        part = builders[name](rng, int(n * share[name]))
        part["label"] = name
        parts.append(part)
    df = pd.concat(parts, ignore_index=True)
    df["failed_logins"] = df["failed_logins"].astype(int)
    df["unique_ports"] = df["unique_ports"].astype(int)
    df["is_attack"] = (df["label"] != "Normal").astype(int)
    start = pd.Timestamp("2026-10-01")
    df["timestamp"] = start + pd.to_timedelta(rng.integers(0, 7 * 24 * 3600, len(df)), unit="s")
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def save_traffic(path="data/traffic.csv", n=20000, seed=42):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = generate_traffic(n, seed)
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    data = save_traffic()
    print(f"Saved data/traffic.csv with {len(data)} rows")
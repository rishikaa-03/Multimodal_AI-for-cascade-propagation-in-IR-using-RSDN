"""
System test -- the whole pipeline, run the way a real deployment would:
raw CSVs -> build_graph.py -> run_simulator.py -> run_models.py ->
run_frontier_detector.py -> network_risk_summary.py -> app.py boot.

This is NOT unit/integration testing (those check code correctness in
isolation and across two modules). This is the acceptance-level check:
"does the actual product work, end to end, the way a user would run it."
Each stage is a real subprocess call to the real runner script, not a
mocked import -- so this also catches issues unit/integration tests can't
(missing files, wrong working directory, CLI-level crashes).
"""
import os
import subprocess
import sys
import time
import json

GRAPH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.normpath(os.path.join(GRAPH_DIR, "..", "data"))

STAGES = [
    ("Module 2 — build_graph.py", [sys.executable, "build_graph.py"]),
    ("Module 3 — run_simulator.py", [sys.executable, "run_simulator.py"]),
    ("Module 4 — run_models.py", [sys.executable, "run_models.py"]),
    ("Module 5 — run_frontier_detector.py", [sys.executable, "run_frontier_detector.py"]),
    ("Module 8 — network_risk_summary.py", [sys.executable, "network_risk_summary.py"]),
]

EXPECTED_ARTIFACTS = [
    "multiplex_graph.pkl",
    "graph_simulation_dataset.csv",
    "cause_classifier.pkl",
    "cascade_predictor.pkl",
    "frontier_alerts_summary.csv",
    "station_risk_summary.csv",
    "edge_risk_summary.csv",
]


def run_stage(name, cmd):
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=GRAPH_DIR, capture_output=True, text=True, timeout=300)
    elapsed = time.perf_counter() - t0
    ok = proc.returncode == 0
    return {
        "stage": name, "ok": ok, "elapsed_s": round(elapsed, 2),
        "returncode": proc.returncode,
        "stderr_tail": proc.stderr[-800:] if not ok else "",
    }


def check_artifacts():
    results = []
    for fname in EXPECTED_ARTIFACTS:
        path = os.path.join(GRAPH_DIR, fname)
        results.append({"artifact": fname, "exists": os.path.exists(path),
                         "size_bytes": os.path.getsize(path) if os.path.exists(path) else 0})
    return results


def check_dashboard_boot():
    import urllib.request
    import urllib.error

    port = 8766
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py",
         "--server.headless", "true", f"--server.port={port}"],
        cwd=GRAPH_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    time.sleep(7)
    try:
        try:
            resp = urllib.request.urlopen(f"http://localhost:{port}", timeout=10)
            http_code = str(resp.getcode())
        except urllib.error.HTTPError as e:
            http_code = str(e.code)
        except Exception as e:
            http_code = f"ERROR: {e}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    return {"http_code": http_code, "ok": http_code == "200"}


def main():
    print("=" * 70)
    print("SYSTEM TEST — full pipeline, run as a real deployment would")
    print("=" * 70)

    report = {"stages": [], "artifacts": [], "dashboard": None}

    for name, cmd in STAGES:
        print(f"\n[RUN] {name}")
        result = run_stage(name, cmd)
        report["stages"].append(result)
        status = "PASS" if result["ok"] else "FAIL"
        print(f"  [{status}] returncode={result['returncode']}  elapsed={result['elapsed_s']}s")
        if not result["ok"]:
            print(f"  stderr: {result['stderr_tail']}")

    print("\n[CHECK] expected artifacts produced")
    artifacts = check_artifacts()
    report["artifacts"] = artifacts
    for a in artifacts:
        status = "PASS" if a["exists"] and a["size_bytes"] > 0 else "FAIL"
        print(f"  [{status}] {a['artifact']}  ({a['size_bytes']} bytes)")

    print("\n[RUN] Module 6 — dashboard boot (app.py)")
    dash = check_dashboard_boot()
    report["dashboard"] = dash
    print(f"  [{'PASS' if dash['ok'] else 'FAIL'}] HTTP {dash['http_code']}")

    all_stages_ok = all(s["ok"] for s in report["stages"])
    all_artifacts_ok = all(a["exists"] and a["size_bytes"] > 0 for a in report["artifacts"])
    overall = all_stages_ok and all_artifacts_ok and dash["ok"]

    print("\n" + "=" * 70)
    print(f"SYSTEM TEST RESULT: {'PASS' if overall else 'FAIL'}")
    print("=" * 70)

    with open(os.path.join(GRAPH_DIR, "tests", "system_test_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()

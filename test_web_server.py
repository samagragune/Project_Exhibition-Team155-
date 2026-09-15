# test_web_server.py
# Tests FastAPI server endpoints using TestClient
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from fastapi.testclient import TestClient
from simulator import app

client = TestClient(app)

print("Testing HTTP endpoints...")

# 1. Test index HTML
res = client.get("/")
assert res.status_code == 200
assert "CYBER-ARM 3D" in res.text
print("  [PASS] GET / (Dashboard HTML loaded)")

# 2. Test capture RGB
res = client.get("/capture")
assert res.status_code == 200
assert res.headers["content-type"] == "image/jpeg"
print(f"  [PASS] GET /capture (RGB image returned: {len(res.content)} bytes)")

# 3. Test capture depth heatmap
res = client.get("/capture_depth")
assert res.status_code == 200
assert res.headers["content-type"] == "image/jpeg"
print(f"  [PASS] GET /capture_depth (Depth heatmap returned: {len(res.content)} bytes)")

# 4. Test telemetry
res = client.get("/telemetry")
assert res.status_code == 200
data = res.json()
assert "end_effector_cm" in data
print(f"  [PASS] GET /telemetry (EE pos: {data['end_effector_cm']}, Scenario: {data['active_scenario']})")

# 5. Test scenarios list
res = client.get("/scenarios")
assert res.status_code == 200
sc_list = res.json()["scenarios"]
assert len(sc_list) == 4
print(f"  [PASS] GET /scenarios (Returned {len(sc_list)} cognitive scenarios)")

# 6. Test switch scenario
res = client.post("/switch_scenario", json={"scenario": "mars_rover"})
assert res.status_code == 200
assert res.json()["active_scenario"] == "mars_rover"
print("  [PASS] POST /switch_scenario (Switched to mars_rover)")

# 7. Test manual jog move
res = client.post("/api/manual_move", json={"action": "move", "x": 22.0, "y": 22.0, "z": 8.0})
assert res.status_code == 200
print(f"  [PASS] POST /api/manual_move (Moved to reached: {res.json()['reached_position_cm']})")

# 8. Test park arm
res = client.post("/execute_tool", json={"tool": "park_arm", "arguments": {"pose": "survey"}})
assert res.status_code == 200
print(f"  [PASS] POST /execute_tool (Park arm executed)")

# 9. Test scan
res = client.post("/api/scan")
assert res.status_code == 200
print(f"  [PASS] POST /api/scan (Scan returned {len(res.json()['objects'])} objects)")

print("\n[ALL HTTP & FASTAPI SERVER TESTS PASSED!]")

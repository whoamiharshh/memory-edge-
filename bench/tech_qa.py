"""Does the app answer the engineering questions it SHOULD answer, from the right source, and decline the rest?

  (start the app first)   .venv\\Scripts\\python.exe -m bench.tech_qa

The questions are taken from the user's own question list (robots, industrial automation, kiosks, vehicles, mobile, maths,
physics, electronics, embedded, AI, safety, security ...). Each has an expected behaviour:

  define     a "what is X" / "how does X work" question. Correct = it answered from the technical library with the entry for
             X (title contains the expected string). Declining counts as a miss in coverage, never as a wrong answer.
  compare    "X vs Y". Correct = both entries are shown, and the answer says it is two definitions, not a comparison.
  decline    design / calculation / what-if / diagnosis / judgment questions. Correct = "Needs internet connection for this."
             and nothing quoted. Answering them with a lead paragraph is a WRONG answer.

accuracy = right / (right + wrong)   - how often a given answer is right.   coverage = right / questions that should be answered.
"""
from __future__ import annotations

import json
import pathlib
import statistics
import time

import httpx

BASE = "http://127.0.0.1:9000/proxy/device/"
ROOT = pathlib.Path(__file__).resolve().parent.parent
NEEDS = "Needs internet connection for this."

DEFINE = [
    ("What is a robot?", "robot"), ("What is an industrial robot?", "industrial robot"),
    ("What is a collaborative robot?", "cobot"), ("What is an end effector?", "end effector"),
    ("What is a degree of freedom?", "degrees of freedom"), ("What is a SCARA robot?", "scara"),
    ("How does a delta robot work?", "delta robot"), ("What is a servo motor?", "servomotor"),
    ("What is backlash?", "backlash"), ("What is forward kinematics?", "forward kinematics"),
    ("What is inverse kinematics?", "inverse kinematics"), ("What is a Jacobian?", "jacobian"),
    ("What are Euler angles?", "euler angles"), ("What is gimbal lock?", "gimbal lock"),
    ("What is PID control?", "pid controller"), ("What is a Kalman filter?", "kalman filter"),
    ("What is SLAM?", "simultaneous localization"), ("What is an encoder?", "rotary encoder"),
    ("What is an IMU?", "inertial measurement unit"), ("How does LiDAR work?", "lidar"),
    ("What is a resolver?", "resolver"), ("What is sensor fusion?", "sensor fusion"),
    ("What is machine vision?", "machine vision"), ("What is a PLC?", "programmable logic controller"),
    ("What is SCADA?", "scada"), ("What is OEE?", "overall equipment effectiveness"),
    ("What is ladder logic?", "ladder logic"), ("What is a watchdog timer?", "watchdog timer"),
    ("What is OPC UA?", "opc unified architecture"), ("What is MQTT?", "mqtt"), ("What is Modbus?", "modbus"),
    ("What is EtherCAT?", "ethercat"), ("What is PROFINET?", "profinet"), ("What is CANopen?", "canopen"),
    ("What is a VFD?", "variable-frequency drive"), ("What is a contactor?", "contactor"),
    ("What is a solenoid valve?", "solenoid valve"), ("What is a digital twin?", "digital twin"),
    ("What is Industry 4.0?", "fourth industrial revolution"), ("What is IIoT?", "industrial internet of things"),
    ("What is lockout/tagout?", "lockout"), ("What is functional safety?", "functional safety"),
    ("What is ISO 26262?", "iso 26262"), ("What is a safety integrity level?", "safety integrity level"),
    ("What is zero trust?", "zero trust"), ("What is ransomware?", "ransomware"),
    ("What is a man-in-the-middle attack?", "man-in-the-middle"), ("What is a hardware security module?", "hardware security module"),
    ("What is TLS?", "transport layer security"), ("What is a kiosk?", "kiosk"), ("What is EMV?", "emv"),
    ("What is idempotency?", "idempotence"), ("What is an ECU?", "electronic control unit"),
    ("What is CAN bus?", "can bus"), ("What is FlexRay?", "flexray"), ("What is UDS?", "unified diagnostic services"),
    ("What is OBD?", "on-board diagnostics"), ("What is V2X?", "vehicle-to-everything"),
    ("What is a battery management system?", "battery management system"), ("What is thermal runaway?", "thermal runaway"),
    ("What is SOC in a battery?", "state of charge"), ("What is ADAS?", "advanced driver-assistance"),
    ("What is adaptive cruise control?", "adaptive cruise control"), ("What is AUTOSAR?", "autosar"),
    ("What is an inverter?", "power inverter"), ("What is regenerative braking?", "regenerative braking"),
    ("What is an SoC?", "system on a chip"), ("What is an NPU?", "neural processing unit"),
    ("What is eSIM?", "esim"), ("What is OLED?", "oled"), ("What is Bluetooth Low Energy?", "bluetooth low energy"),
    ("What is NFC?", "near-field communication"), ("What is MIMO?", "mimo"), ("What is beamforming?", "beamforming"),
    ("What is an eigenvalue?", "eigenvalues"), ("What is the Laplace transform?", "laplace transform"),
    ("What is a transfer function?", "transfer function"), ("What is aliasing?", "aliasing"),
    ("What is a MOSFET?", "mosfet"), ("What is an IGBT?", "insulated-gate bipolar transistor"),
    ("What is a buck converter?", "buck converter"), ("What is an operational amplifier?", "operational amplifier"),
    ("What is an RTOS?", "real-time operating system"), ("What is priority inversion?", "priority inversion"),
    ("What is DMA?", "direct memory access"), ("What is JTAG?", "jtag"), ("What is a mutex?", "lock"),
    ("What is overfitting?", "overfitting"), ("What is a confusion matrix?", "confusion matrix"),
    ("What is reinforcement learning?", "reinforcement learning"), ("What is edge computing?", "edge computing"),
    ("What is MTBF?", "mean time between failures"), ("What is FMEA?", "failure mode and effects analysis"),
    ("What is predictive maintenance?", "predictive maintenance"), ("What is a bathtub curve?", "bathtub curve"),
    ("What is Six Sigma?", "six sigma"), ("What is an IP rating?", "ip code"),
]
COMPARE = [("PLC vs DCS?", ["programmable logic controller", "distributed control system"]),
           ("TCP vs UDP?", ["transmission control protocol", "user datagram protocol"]),
           ("Servo vs stepper motor?", ["servomotor", "stepper motor"]),
           ("LiDAR vs radar?", ["lidar", "radar"]), ("Bluetooth vs NFC?", ["bluetooth", "near-field communication"]),
           ("OLED vs LCD?", ["oled", "liquid-crystal display"])]
DECLINE = [
    "Design a robot cell that safely works beside humans.", "Design a PLC-controlled packaging line.",
    "Design a kiosk capable of operating during internet outages.", "Design an EV battery-management system.",
    "How would you design redundancy without doubling the cost?", "How would you detect an actuator that is slowly degrading?",
    "What if the PLC loses power?", "What if a kiosk loses internet?", "What happens if the network disappears?",
    "What happens if two sensors disagree?", "Calculate robot joint torque.", "Calculate OEE for a line running 7 hours.",
    "Calculate EV charging time.", "Diagnose a robot that suddenly stops.", "Why use PID?",
    "Why do robots use feedback?", "Which is better, SCARA or delta, for high-speed sorting?",
    "How do you prove a design is safe?", "How do you reduce cost by 30 percent?", "How would you redesign it from scratch?",
]


def ask(c: httpx.Client, q: str) -> tuple[dict, float]:
    t = time.perf_counter()
    r = c.post(BASE + "ask", json={"text": q}).json()
    return r, (time.perf_counter() - t) * 1000


def main() -> None:
    c = httpx.Client(timeout=300)
    rows, right, wrong, missed = [], 0, 0, 0
    for q, key in DEFINE:
        r, ms = ask(c, q)
        titles = " | ".join((u.get("title") or "").lower() for u in r.get("used", []))
        answered = r.get("mode") in ("quoted", "llm")
        ok = answered and key in titles
        right, wrong, missed = right + ok, wrong + (answered and not ok), missed + (not answered)
        rows.append({"kind": "define", "q": q, "ok": ok, "answered": answered, "titles": titles, "ms": round(ms)})
    for q, keys in COMPARE:
        r, ms = ask(c, q)
        titles = " | ".join((u.get("title") or "").lower() for u in r.get("used", []))
        answered = r.get("mode") in ("quoted", "llm")
        ok = answered and all(k in titles for k in keys) and "side-by-side" in r["answer"]
        right, wrong, missed = right + ok, wrong + (answered and not ok), missed + (not answered)
        rows.append({"kind": "compare", "q": q, "ok": ok, "answered": answered, "titles": titles, "ms": round(ms)})
    declined_right = declined_wrong = 0
    for q in DECLINE:
        r, ms = ask(c, q)
        ok = r["answer"].strip().startswith(NEEDS) and not r.get("used")
        declined_right, declined_wrong = declined_right + ok, declined_wrong + (not ok)
        rows.append({"kind": "decline", "q": q, "ok": ok, "answered": not ok, "titles": r["answer"][:120], "ms": round(ms)})
    summary = {"should_answer": len(DEFINE) + len(COMPARE), "right": right, "wrong": wrong, "declined_by_mistake": missed,
               "should_decline": len(DECLINE), "declined_correctly": declined_right,
               "answered_when_it_should_decline": declined_wrong,
               "accuracy_pct": round(100 * right / max(1, right + wrong + declined_wrong), 1),
               "coverage_pct": round(100 * right / (len(DEFINE) + len(COMPARE)), 1),
               "latency_ms_median": int(statistics.median(r["ms"] for r in rows)),
               "latency_ms_max": max(r["ms"] for r in rows)}
    for r in rows:
        if r["kind"] == "decline" and not r["ok"]:
            print("ANSWERED BUT SHOULD DECLINE |", r["q"], "->", r["titles"])
        elif r["kind"] != "decline" and r["answered"] and not r["ok"]:
            print("WRONG ENTRY |", r["q"], "->", r["titles"])
        elif r["kind"] != "decline" and not r["answered"]:
            print("MISSED |", r["q"])
    print(json.dumps(summary, indent=1))
    (ROOT / "bench" / "results").mkdir(exist_ok=True)
    (ROOT / "bench" / "results" / "tech_qa.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1,
                                                                        ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

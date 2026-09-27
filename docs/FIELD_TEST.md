# Field test: real devices, a real sensor, a real technician

Everything before this was measured on one laptop with recorded data. This guide is how to test the three things
that are **not verified yet**:

1. several **physical** devices talking to the cloud over a real network (not one laptop),
2. a **real sensor on a real machine** with a real fault and a real fix,
3. whether a **real technician** finds the workflow usable.

Write down what happens, including failures; a failed step is a finding, not a mistake.

## What runs where (supported platforms, verified from PyPI for `qdrant-edge-py` 0.8.0)

| Hardware | Can run a full edge device? | How |
|---|---|---|
| Windows 10/11 PC or laptop (x64) | **Yes** (tested) | this repo, `.venv` |
| Mac (Intel or Apple Silicon) | Yes (wheel exists; **not tested by us**) | same Python steps |
| Linux PC, Raspberry Pi 4/5 (64-bit OS), Jetson (ARM64) | Yes (wheels for x64 + ARM64; **not tested by us**) | same Python steps |
| Android / iPhone | **No** (no wheel for phones) | phone = **sensor + screen** over HTTPS (`/sensor` page) |
| Windows on ARM | No wheel | use x64 emulation or another device |

## Part 1: two real computers + the cloud (about 30 min)

You need: laptop **L** (cloud + Qdrant Server), a second computer **P** (edge device), same Wi-Fi.

1. On **L**, create the HTTPS certificates (they include L's current Wi-Fi IP):
   `.venv\Scripts\python.exe tools\make_certs.py`
2. On **L**, start Qdrant Server and the cloud on the network, with TLS:
   start `qdrant_server\qdrant.exe` (as in `demo\run_demo.ps1`), then
   `.venv\Scripts\python.exe -m cloud.main --host 0.0.0.0 --port 8100 --tls --bootstrap`
   Windows will ask to allow the firewall: allow **private networks** only.
3. Copy `runtime\tls\ca.pem` and one device token from `runtime\cloud\bootstrap.json` to **P** (USB stick).
   Never copy `ca.key` or `tokens.json`.
4. On **P** (repo installed as in README "Run it"):
   `.venv\Scripts\python.exe -m edge.main --name devP --site site9 --port 8101 --cloud https://<L's IP>:8100 --device-token <token> --operator-token <choose> --ca ca.pem --fit-baseline`
5. Open `http://127.0.0.1:8101` on P, replay a fault, record an action, replay healthy, confirm → SHARE.
6. Check on L's fleet UI (`https://<L's IP>:8100`) that the case arrived. Then **pull the Wi-Fi plug on P**, do another
   episode, plug back in: it must arrive exactly once.

Record: did the push arrive? how long after reconnecting? any certificate errors? (the refusal of an untrusted
certificate is correct behaviour; test it once by starting P **without** `--ca`).

## Part 2: a real machine, a real fault, a real fix (phone as the sensor, about 45 min)

Machine: a **desk or pedestal fan** (safe, cheap, has a real rotor). Fault: **imbalance** from a coin taped to one
blade. Fix: remove the coin. This is the classic 1x imbalance fault; the phone profile's physics rule targets it.

1. On the laptop that runs the device, with certificates made (Part 1 step 1):
   `.venv\Scripts\python.exe -m edge.main --name fan1 --site home --host 0.0.0.0 --port 8101 --tls --profile lowrate-accel --component fan --operator-token <choose> --no-sync`
2. On the phone (same Wi-Fi), open `https://<laptop IP>:8101/sensor`. The browser warns about the certificate:
   either continue (still encrypted, but not verified) or install `ca.pem` on the phone once (verified).
3. Put the phone **flat on the fan's base or motor housing** (not the moving guard), fan at a fixed speed.
   Enter the operator token, **Start sensor** (allow motion access), then **Capture healthy baseline** and wait for
   "baseline fitted" (~1.5 min, nobody touching the table).
4. Switch the fan off, tape a coin (or a small weight) near the tip of one blade, switch on, same speed.
   Expected: state **NEW** then **MERGE**; on the laptop UI an episode with hint **imbalance** and "shaft ≈ … Hz".
   Note the shaft Hz: with a 60 Hz phone the device can only judge 2x if 2 × shaft is below 30 Hz; the UI says
   so explicitly when it is not.
5. On the laptop UI: confirm fault class *imbalance*, record action *rebalance*, remove the coin, fan on.
   Expected: verification fills to 20/20 → **symptom resolved**; click *Worked* → **SHARE**.
6. Repeat once with the fan at a different speed **without** the coin. If it opens an episode, that is a new healthy
   state: press **Not a fault: normal operation** and check that the same speed is afterwards NORMAL.

Record: shaft Hz, how many windows until NEW, verification windows, any false episodes, phone model and browser.
Photos of the setup help the README.

## Part 3: a real technician (about 30 min per person)

Who: a mechanic at a bike/car service centre, a college workshop technician, a maintenance person at any plant.
Show Part 2 (or the demo), then let **them** drive the device UI. Do not help unless they are stuck > 1 minute.

Ask, and write down the answers word for word:
1. Tasks they complete without help: open the episode, write a note, pick the fault class, record the action,
   confirm the outcome, read the fleet evidence. (✓ / needed help / failed)
2. "What does *symptom resolved for 20 windows* mean to you?" (do they read it as "fixed for sure"? it isn't)
3. "Would you trust a fix that worked at another site? What would you need to see?"
4. "Is anything on this screen useless or confusing?"
5. "Would you write notes on a phone/tablet next to the machine? In which language?"
6. "What stops you from using something like this at work?"
7. Score 1-5: *I could use this without training.* / *The evidence shown helps me decide.*

Put the answers in a table in `docs/FIELD_NOTES.md` (who: role only, never names). This is the practitioner
validation RESEARCH.md Part R names as the biggest product risk.

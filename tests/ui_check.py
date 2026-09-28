"""Headless UI check against the RUNNING demo (demo/run_demo.ps1): signs in to the device and fleet UIs, waits for
real content, fails on any console error, writes screenshots to runtime/screenshots/.

  .venv\\Scripts\\python.exe tests\\ui_check.py
Uses the Microsoft Edge that ships with Windows (Playwright channel "msedge"); no browser download.
"""
import json
import pathlib
import sys

from playwright.sync_api import sync_playwright

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "runtime" / "screenshots"
BOOT = json.loads((ROOT / "runtime" / "cloud" / "bootstrap.json").read_text())


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    errors, report = [], {}
    with sync_playwright() as p:
        br = p.chromium.launch(channel="msedge", headless=True)
        # bypass_csp lets the TEST's own wait expressions run; the pages' CSP still applies to the app's scripts
        pg = br.new_page(viewport={"width": 1440, "height": 1900}, bypass_csp=True)
        pg.on("console", lambda m: m.type == "error" and errors.append(f"{pg.url}: {m.text}"))
        pg.on("pageerror", lambda e: errors.append(f"{pg.url}: {e}"))

        pg.goto("http://127.0.0.1:8101/")
        pg.fill("#tok", "operator-devA")
        pg.click("#loginForm button[type=submit]")
        pg.wait_for_function("document.querySelectorAll('#epList .item').length > 0", timeout=15000)
        pg.wait_for_function("document.querySelectorAll('#gates li').length > 0", timeout=15000)
        pg.click("#simBtn")
        pg.wait_for_function("document.querySelector('#sInfo').textContent.includes('ms')", timeout=15000)
        pg.wait_for_timeout(2500)
        report["device"] = pg.evaluate("""() => ({
            episodes: document.querySelectorAll('#epList .item').length,
            gates: [...document.querySelectorAll('#gates li')].map(l => l.textContent),
            outbox_rows: document.querySelectorAll('#obBody tr').length,
            activity: document.querySelectorAll('#feed div').length,
            search: document.querySelector('#sInfo').textContent,
            network: document.getElementById('netLabel').textContent })""")
        pg.screenshot(path=str(OUT / "device_a.png"), full_page=True)

        pg.goto("http://127.0.0.1:8100/")
        pg.fill("#tok", BOOT["admin"])
        pg.click("#loginForm button[type=submit]")
        pg.wait_for_function("document.querySelectorAll('#cases .item').length > 0", timeout=15000)
        pg.click("#cases .item")
        pg.wait_for_function("document.querySelectorAll('#detail table').length > 0", timeout=15000)
        pg.wait_for_timeout(1000)
        report["fleet"] = pg.evaluate("""() => ({
            cases: document.querySelectorAll('#cases .item').length,
            detail_tables: document.querySelectorAll('#detail table').length,
            devices: document.querySelectorAll('#devBody tr').length })""")
        pg.screenshot(path=str(OUT / "fleet.png"), full_page=True)
        br.close()
    report["console_errors"] = errors
    print(json.dumps(report, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())

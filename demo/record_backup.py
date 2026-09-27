"""Backup demo recording: runs the scripted scenario (demo/scenario.py) against the RUNNING system while a headless
Microsoft Edge records the real UIs, switching between Device A, the fleet cloud and Device B as the steps advance,
with a caption bar naming the current step. Nothing is simulated in the video: it is the live system.

  powershell -ExecutionPolicy Bypass -File demo\\run_demo.ps1 -Reset      (fresh state is required)
  .venv\\Scripts\\python.exe -m demo.record_backup
Output: runtime/recording/backup_demo.webm (+ a PNG per step). Plays in any browser or VLC.
"""
from __future__ import annotations

import json
import pathlib
import shutil
import threading
import time

from playwright.sync_api import sync_playwright

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "runtime" / "recording"
BOOT = json.loads((ROOT / "runtime" / "cloud" / "bootstrap.json").read_text())
PAGES = {"A": ("http://127.0.0.1:8101/", "operator-devA"), "B": ("http://127.0.0.1:8102/", "operator-devB"),
         "cloud": ("http://127.0.0.1:8100/", BOOT["admin"])}
# which UI to show during each scenario step, and where to scroll it
SHOW = {1: ("A", "#chart"), 2: ("A", "#gates"), 3: ("A", "#gates"), 4: ("A", "#chart"), 5: ("A", "#obBody"),
        6: ("cloud", "#cases"), 7: ("B", "#chart"), 8: ("B", "#sFleet"), 9: ("A", "#feed")}
REPLAY_INTERVAL = 0.06                    # slow the replay down so the chart visibly moves on video


def caption(pg, text: str) -> None:
    pg.evaluate("""(t) => {
        let c = document.getElementById('rec-caption');
        if (!c) { c = document.createElement('div'); c.id = 'rec-caption';
          c.style.cssText = 'position:fixed;left:0;right:0;bottom:0;z-index:99999;padding:10px 18px;' +
            'font:600 18px system-ui;color:#fff;background:rgba(10,14,20,.88);border-top:2px solid #5aa9ff';
          document.body.append(c); }
        c.textContent = t; }""", text)


def main() -> None:
    import demo.scenario as sc
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)
    state = {"step": 0, "title": "", "done": False, "error": None}
    shown_ev = threading.Event()
    orig_step = sc.step

    def step(title):
        orig_step(title)
        shown_ev.clear()
        state.update(step=sc.STEP[0], title=title)
        shown_ev.wait(timeout=90)         # the recorder has switched page + done its UI actions for this step
        time.sleep(2.0)                   # time to read the caption before the step runs

    def play(c, fid, n, start=0):
        sc.call(c, "POST", "/api/replay", {"fid": fid, "n": n, "interval": REPLAY_INTERVAL, "start": start})
        while sc.call(c, "GET", "/api/stats")["replay"]["playing"]:
            time.sleep(0.2)

    sc.step, sc.play = step, play         # scenario.main() looks both names up at call time

    def run():
        try:
            sc.main()
        except BaseException as e:        # SystemExit from a failed check too
            state["error"] = repr(e)
        finally:
            state["done"] = True

    with sync_playwright() as p:
        br = p.chromium.launch(channel="msedge", headless=True)
        ctx = br.new_context(viewport={"width": 1440, "height": 900}, record_video_dir=str(OUT),
                             record_video_size={"width": 1440, "height": 900})
        pg = ctx.new_page()
        for key, (url, tok) in PAGES.items():            # sign in once per origin (token kept per tab + origin)
            pg.goto(url)
            pg.fill("#tok", tok)
            pg.click("#loginForm button[type=submit]")
            pg.wait_for_timeout(800)
        pg.goto(PAGES["A"][0])
        caption(pg, "Machine Memory at the Edge - live system: Qdrant Server + fleet cloud + two edge devices")
        pg.wait_for_timeout(3000)
        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        shown = 0
        while not state["done"]:
            if state["step"] != shown:
                shown = state["step"]
                where, anchor = SHOW.get(shown, ("A", "#chart"))
                pg.goto(PAGES[where][0])
                pg.wait_for_timeout(1200)
                try:
                    pg.locator(anchor).first.scroll_into_view_if_needed(timeout=3000)
                except Exception:
                    pass
                caption(pg, f"Step {shown}: {state['title']}")
                try:
                    if where == "cloud":
                        pg.wait_for_timeout(2500)             # cases appear after step 5's push
                        pg.click("#cases .item", timeout=8000)
                    if shown == 8:                            # B is offline with its episode: show it in the UI
                        pg.click("#simBtn", timeout=8000)
                        pg.wait_for_function("document.querySelector('#sInfo').textContent.includes('ms')", timeout=20000)
                        pg.locator("#sFleet").scroll_into_view_if_needed()
                        pg.wait_for_timeout(3500)
                        pg.click("#briefBtn", timeout=8000)
                        pg.wait_for_function("document.querySelector('#briefOut .brief') !== null", timeout=90000)
                        pg.locator("#briefOut").scroll_into_view_if_needed()
                        pg.wait_for_timeout(4000)
                except Exception as e:
                    print("recorder: UI action skipped:", repr(e)[:120])
                pg.screenshot(path=str(OUT / f"step{shown}.png"))
                shown_ev.set()
            pg.wait_for_timeout(400)
        final = "ALL STEPS PASSED (live, asserted)" if not state["error"] else f"STOPPED: {state['error']}"
        caption(pg, final)
        pg.wait_for_timeout(3000)
        video = pg.video.path()
        ctx.close()
        br.close()
    pathlib.Path(video).replace(OUT / "backup_demo.webm")
    print(final, "->", OUT / "backup_demo.webm")


if __name__ == "__main__":
    main()

"""
Record an animated GIF walkthrough of the Streamlit dashboard.

Usage (from the project root):

    # 1. one-time setup (dev only, not in requirements.txt)
    pip install playwright pillow
    playwright install chromium

    # 2. start the dashboard in another terminal
    streamlit run app/dashboard.py

    # 3. record
    python scripts/record_dashboard_gif.py

Output: docs/dashboard_demo.gif

How it works, step by step:
1. Playwright opens the dashboard in a headless Chromium browser.
2. The script clicks through each tab, fills in inputs and scrolls,
   taking a screenshot after every step.
3. Pillow stitches the screenshots into one looping GIF. Each frame has its
   own display time, so important screens stay visible longer.
"""

import argparse
import io
import tempfile
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_ROWS = """Machine ID,Type,Air temperature [K],Process temperature [K],Rotational speed [rpm],Torque [Nm],Tool wear [min]
M-001,L,298.1,308.6,1551,42.8,0
M-002,M,302.5,310.9,1320,62.0,215
M-003,H,299.0,309.5,1410,55.3,180
M-004,L,300.8,310.2,2650,8.5,40
M-005,L,297.4,308.1,1495,39.1,95
M-006,M,303.1,311.0,1290,68.2,230
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8501")
    parser.add_argument("--out", default=str(ROOT / "docs/dashboard_demo.gif"))
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=760)
    parser.add_argument("--scale", type=float, default=0.75,
                        help="resize factor for the GIF (smaller = lighter)")
    args = parser.parse_args()

    frames = []  # list of (PIL image, duration in ms)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(
            viewport={"width": args.width, "height": args.height})

        def snap(ms=1200, wait=700):
            """Wait for Streamlit to finish re-running, then screenshot."""
            page.wait_for_timeout(wait)
            # Streamlit shows a "running" widget and greys out stale
            # elements while it re-runs; wait until both are gone.
            page.wait_for_function(
                "!document.querySelector('[data-testid=\"stStatusWidget\"]')"
                " && !document.querySelector('[data-stale=\"true\"]')",
                timeout=30000)
            page.wait_for_timeout(300)
            img = Image.open(io.BytesIO(page.screenshot()))
            frames.append((img.convert("RGB"), ms))

        def scroll(dy, ms=1200):
            page.mouse.move(args.width * 0.6, args.height * 0.6)
            page.mouse.wheel(0, dy)
            snap(ms)

        def tab(name):
            page.get_by_role("tab", name=name).click()
            page.wait_for_timeout(2500)
            # back to the top of the page
            page.mouse.move(args.width * 0.6, args.height * 0.6)
            page.mouse.wheel(0, -5000)

        def set_number(label, value):
            box = page.get_by_label(label, exact=True)
            box.fill(str(value))
            box.press("Enter")

        page.goto(args.url)
        page.wait_for_selector("text=Frozen test-set results", timeout=60000)
        page.wait_for_timeout(2500)

        # --- 1. Overview ---
        snap(2500)
        scroll(450, 2500)

        # --- 2. Predict a machine: healthy, then worn & overloaded ---
        tab("🔧 Predict a machine")
        snap(1500)
        scroll(380, 2000)
        page.mouse.wheel(0, -5000)
        set_number("Tool wear [min]", 220)
        snap(700)
        set_number("Torque [Nm]", 62)
        snap(1000)
        scroll(380, 3000)

        # --- 3. Batch scoring ---
        tab("📁 Batch scoring")
        snap(1200)
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "machines.csv"
            csv_path.write_text(SAMPLE_ROWS)
            page.locator("input[type=file]").set_input_files(str(csv_path))
            page.wait_for_selector("text=Machines scored", timeout=30000)
        snap(1500, wait=1500)
        scroll(300, 3000)

        # --- 4. Model insights ---
        tab("🧠 Model insights")
        snap(2200, wait=1500)
        scroll(650, 2500)
        scroll(500, 2500)

        # --- 5. Data exploration ---
        tab("🔍 Data exploration")
        snap(2000, wait=1500)
        scroll(500, 2000)

        browser.close()

    # --- stitch the GIF ---
    size = (int(args.width * args.scale), int(args.height * args.scale))
    images = [img.resize(size, Image.LANCZOS) for img, _ in frames]
    durations = [ms for _, ms in frames]
    # Each frame gets its own 256-colour palette so chart colours stay true
    gif_frames = [im.quantize(colors=256, method=Image.MEDIANCUT,
                              dither=Image.NONE) for im in images]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    gif_frames[0].save(out, save_all=True, append_images=gif_frames[1:],
                       duration=durations, loop=0, optimize=True)
    print(f"Saved {len(frames)} frames to {out} "
          f"({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

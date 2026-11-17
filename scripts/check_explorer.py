"""Exercise the ranking explorer in a fresh desktop and mobile Chromium page."""

import argparse
import os
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


def check_explorer(base, screenshot=None):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=os.environ.get("CHROMIUM_EXECUTABLE"))
        page = browser.new_page(viewport={"width": 1280, "height": 960}, device_scale_factor=1)
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(base, wait_until="domcontentloaded")
        status = page.get_by_role("status")
        panels = page.locator("#results article")
        expect(status).to_have_text(
            "Results for “grind beans while camping without electricity”", timeout=45000
        )
        expect(panels).to_have_count(2)
        expect(page.locator("#catalog")).to_have_text("30 products indexed")
        expect(panels.last.get_by_text("8 scored by BGE", exact=False)).to_be_visible()
        if screenshot:
            page.screenshot(path=screenshot)
        panels.last.get_by_text("Stage timings", exact=True).click()
        expect(panels.last.get_by_text("Query encoding", exact=True)).to_be_visible()
        panels.last.get_by_text("Ranking details", exact=True).first.click()
        expect(panels.last.get_by_text("BGE logit", exact=True).first).to_be_visible()

        page.get_by_label("Product query").fill("DESK-ARM")
        page.get_by_label("Compare with").select_option("rerank-full")
        page.get_by_role("button", name="Search").click()
        expect(status).to_have_text("Results for “DESK-ARM”", timeout=45000)
        for panel in panels.all():
            expect(panel.get_by_role("heading", level=3).first).to_have_text(
                "Gas-spring monitor arm"
            )
        expect(panels.last.get_by_text("30 scored by BGE", exact=False)).to_be_visible()

        page.set_viewport_size({"width": 375, "height": 760})
        page.get_by_label("In stock only").check()
        page.get_by_label("Max price (USD)").fill("0")
        page.get_by_label("Compare with").select_option("")
        page.get_by_role("button", name="Search").click()
        expect(panels).to_have_count(1)
        expect(panels.get_by_text("No matching products.", exact=False)).to_be_visible()
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")

        page.route("**/search", lambda route: route.fulfill(status=503, body="unavailable"))
        page.get_by_role("button", name="Search").click()
        expect(status).to_contain_text("temporarily unavailable")
        expect(panels).to_have_count(0)
        assert not errors, errors
        browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base")
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args()
    check_explorer(args.base, args.screenshot)

import os, json, re
from pathlib import Path

import requests
import pandas as pd
from dotenv import load_dotenv


GEMINI_MODEL = "gemini-3.6-flash"

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def ask_ai(prompt):
    """Try Gemini first, then Ollama. Returns text, or None if both fail."""
    
    key = os.getenv("GEMINI_API_KEY")
    if key:
        response = None
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
            response = requests.post(
                url,
                headers={"x-goog-api-key": key},
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=30,
            )
            response.raise_for_status()
            return response.json()["candidates"][0]["content"]["parts"][0]["text"]
        except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as e:
            print(f"Gemini failed; trying Ollama: {e}")
            if response is not None:
                print("Gemini response:", response.text)
    else:
        print("GEMINI_API_KEY is not set; skipping Gemini.")

    # 2. Ollama (local)
    try:
        r = requests.post("http://localhost:11434/api/generate",
                          json={"model": "qwen2.5:3b", "prompt": prompt, "stream": False},
                          timeout=60)
        return r.json()["response"]
    except Exception as e:
        print("Ollama error:", e)
        return None


def parse_alert(text):
    """Alert text -> {"start": ..., "end": ...} or None."""
    # Step 1: ask the AI to turn text into JSON
    answer = ask_ai('Does this alert describe a battery outage? Reply with JSON only: '
                    '{"outage": true or false, "start": "YYYY-MM-DD HH:MM:SS", '
                    '"end": "YYYY-MM-DD HH:MM:SS"}. Data is from January 2026. Alert: ' + text)
    if answer:
        match = re.search(r"\{.*\}", answer, re.S)
        if match:
            try:
                data = json.loads(match.group())
                if data.get("outage"):
                    return {"start": data["start"], "end": data["end"]}
                return None
            except Exception:
                pass

    # Step 2: AI not available -> simple keyword search
    if not re.search(r"battery|outage|maintenance", text, re.I):
        return None
    day = re.search(r"jan\w*\s*(\d+)", text, re.I)
    days = re.search(r"(\d+)\s*day", text, re.I)
    start = pd.Timestamp(f"2026-01-{int(day.group(1)) if day else 10:02d}")
    end = start + pd.Timedelta(days=int(days.group(1)) if days else 1)
    return {"start": start.strftime("%Y-%m-%d %H:%M:%S"),
            "end": end.strftime("%Y-%m-%d %H:%M:%S")}


def explain(row):
    """One dispatch row -> plain English."""
    surplus = row['generation'] - row['demand']
    facts = (
        f"Time: {row['time']}. Price: {row['price']:.0f}. "
        f"Generation: {row['generation']:.0f} MW, demand: {row['demand']:.0f} MW, "
        f"difference (generation minus demand): {surplus:.0f} MW. "
        f"Action requested by controller: {row['reason']}. "
        f"Actually charged batteries: {row['charged']:.0f} MW, "
        f"actually discharged: {row['discharge']:.0f} MW, "
        f"bought from grid: {row['bought']:.0f} MW, sold to grid: {row['sold']:.0f} MW. "
        f"Stored energy now: B1 {row['B1_soc']:.0f} MWh (max 100), B2 {row['B2_soc']:.0f} MWh (max 200)."
    )
    answer = ask_ai(
        "Explain this energy decision in 2 simple sentences. "
        "Use ONLY the facts below. Do not invent causes or numbers. "
        "The Action is what the controller requested; the Charged, Discharged, Bought and Sold "
        "numbers show what actually happened. "
        "If Discharged is 0 during a deficit, the shortfall was bought from the grid (see Bought). "
        "Never say demand went unserved. "
        "Battery power is in MW, stored energy is in MWh. "
        "Reply with only the 2 sentences, no introduction. Facts: " + facts
    )
    return answer or "[template] " + facts

def parse_storm(text, current=None):
    """Storm text -> {"start","end"[,"cancelled_at"]} or None.
    'storm on Jan 12 from 18:00 to 22:00' -> new storm
    'stop at 19:30'                       -> cancels the current storm
    'extended until 01:00'                -> moves the end of the current storm
    """
    t = text.lower()
    times = re.findall(r"(\d{1,2}):(\d{2})\s*(am|pm)?", t)
    day = re.search(r"jan\w*\s*(\d{1,2})", t)
    fmt = lambda x: x.strftime("%Y-%m-%d %H:%M:%S")

    base = pd.Timestamp(current["start"]).normalize() if current else pd.Timestamp("2026-01-12")
    if day:
        base = pd.Timestamp(f"2026-01-{int(day.group(1)):02d}")

    def at(i):
        h, m, ap = times[i]
        h = int(h)
        if ap == "pm" and h < 12:
            h += 12
        if ap == "am" and h == 12:
            h = 0
        return base + pd.Timedelta(hours=h, minutes=int(m))

    if current and times and re.search(r"stop|cancel|over|all clear", t):
        stop = at(0)
        if stop <= pd.Timestamp(current["start"]):
            return current              # stop is before the storm started, ignore it
        out = dict(current)
        out["cancelled_at"] = fmt(stop)
        return out

    if current and times and re.search(r"extend|until|longer|prolong", t):
        out = dict(current)
        out.pop("cancelled_at", None)
        end = at(0)
        if end <= pd.Timestamp(current["start"]):
            end += pd.Timedelta(days=1)          # '01:00' means next day
        out["end"] = fmt(end)
        return out

    if "storm" in t and len(times) >= 2:
        s, e = at(0), at(1)
        if e <= s:
            e += pd.Timedelta(days=1)
        return {"start": fmt(s), "end": fmt(e)}

    return None
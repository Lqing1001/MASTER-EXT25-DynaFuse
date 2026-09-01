from __future__ import annotations

import argparse
import io
import json
import re
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


INDEX_ID = "000300"
BASE_URL = "https://vip.stock.finance.sina.com.cn/corp/view/"
LATEST_URL = BASE_URL + "vII_NewestComponent.php?page={page}&indexid=000300"
HISTORY_URL = BASE_URL + "vII_HistoryComponent.php?page={page}&indexid=000300"
COL_CODE = "\u54c1\u79cd\u4ee3\u7801"
COL_NAME = "\u54c1\u79cd\u540d\u79f0"
COL_IN = "\u7eb3\u5165\u65e5\u671f"
COL_OUT = "\u5254\u9664\u65e5\u671f"


def fetch_html(url: str, retries: int = 3) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/131.0 Safari/537.36"
        ),
        "Referer": "https://vip.stock.finance.sina.com.cn/",
    }
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = response.read()
                declared = response.headers.get_content_charset()
            candidates = [declared, "utf-8", "gb18030"]
            for encoding in dict.fromkeys(item for item in candidates if item):
                try:
                    return raw.decode(encoding)
                except UnicodeDecodeError:
                    continue
            return raw.decode("gb18030", errors="replace")
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(attempt)
    raise RuntimeError(f"Failed to fetch {url}: {last_error}")


def find_page_count(html: str) -> int:
    match = re.search("\u5171\\s*(\\d+)\\s*\u9875", html)
    if not match:
        raise ValueError("Could not find total page count")
    return int(match.group(1))


def select_component_table(html: str, expected_columns: list[str]) -> pd.DataFrame:
    expected = set(expected_columns)
    for table in pd.read_html(io.StringIO(html), displayed_only=False):
        columns = {str(column).strip() for column in table.columns}
        if expected.issubset(columns):
            return table.copy()
    raise ValueError(f"Component table not found; expected={expected_columns}")


def normalize_code(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    digits = re.sub(r"\D", "", text)
    return digits.zfill(6) if digits else ""


def normalize_date(value: object) -> str:
    text = str(value).strip()
    if not text or text in {"--", "nan", "NaT", "None"}:
        return ""
    parsed = pd.to_datetime(text, errors="coerce")
    return text if pd.isna(parsed) else parsed.strftime("%Y-%m-%d")


def scrape_table(kind: str, template: str) -> tuple[pd.DataFrame, int]:
    first_html = fetch_html(template.format(page=1))
    page_count = find_page_count(first_html)
    frames: list[pd.DataFrame] = []
    for page in range(1, page_count + 1):
        html = first_html if page == 1 else fetch_html(template.format(page=page))
        expected = [COL_CODE, COL_NAME, COL_IN]
        if kind == "history":
            expected.append(COL_OUT)
        table = select_component_table(html, expected).rename(
            columns={
                COL_CODE: "stock_code",
                COL_NAME: "stock_name",
                COL_IN: "inclusion_date",
                COL_OUT: "exclusion_date",
            }
        )
        table["stock_code"] = table["stock_code"].map(normalize_code)
        table["stock_name"] = table["stock_name"].astype(str).str.strip()
        table["inclusion_date"] = table["inclusion_date"].map(normalize_date)
        if "exclusion_date" not in table.columns:
            table["exclusion_date"] = ""
        else:
            table["exclusion_date"] = table["exclusion_date"].map(normalize_date)
        table = table[table["stock_code"].str.fullmatch(r"\d{6}", na=False)].copy()
        table.insert(0, "index_code", INDEX_ID)
        table.insert(1, "source_type", f"sina_{kind}")
        table.insert(2, "source_page", page)
        table["source_url"] = template.format(page=page)
        frames.append(table[[
            "index_code", "source_type", "source_page", "stock_code",
            "stock_name", "inclusion_date", "exclusion_date", "source_url",
        ]])
    return pd.concat(frames, ignore_index=True), page_count


def nullable_min(series: pd.Series) -> str | None:
    values = series[series.ne("")]
    return None if values.empty else str(values.min())


def nullable_max(series: pd.Series) -> str | None:
    values = series[series.ne("")]
    return None if values.empty else str(values.max())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    latest, latest_pages = scrape_table("latest", LATEST_URL)
    history, history_pages = scrape_table("history", HISTORY_URL)
    latest.to_csv(args.output_dir / "sina_csi300_latest.csv", index=False, encoding="utf-8-sig")
    history.to_csv(args.output_dir / "sina_csi300_history.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "scraped_at_utc": datetime.now(timezone.utc).isoformat(),
        "single_threaded": True,
        "index_code": INDEX_ID,
        "latest": {
            "pages": latest_pages,
            "rows": int(len(latest)),
            "unique_codes": int(latest["stock_code"].nunique()),
            "min_inclusion_date": nullable_min(latest["inclusion_date"]),
            "max_inclusion_date": nullable_max(latest["inclusion_date"]),
            "url_template": LATEST_URL,
        },
        "history": {
            "pages": history_pages,
            "rows": int(len(history)),
            "unique_codes": int(history["stock_code"].nunique()),
            "min_inclusion_date": nullable_min(history["inclusion_date"]),
            "max_inclusion_date": nullable_max(history["inclusion_date"]),
            "max_exclusion_date": nullable_max(history["exclusion_date"]),
            "open_ended_rows": int(history["exclusion_date"].eq("").sum()),
            "url_template": HISTORY_URL,
        },
    }
    (args.output_dir / "sina_csi300_scrape_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

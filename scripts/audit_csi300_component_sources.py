from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


COL_DATE = "\u65e5\u671fDate"
COL_INDEX_CODE = "\u6307\u6570\u4ee3\u7801 Index Code"
COL_INDEX_NAME = "\u6307\u6570\u540d\u79f0 Index Name"
COL_STOCK_CODE = "\u6210\u4efd\u5238\u4ee3\u7801Constituent Code"
COL_STOCK_NAME = "\u6210\u4efd\u5238\u540d\u79f0Constituent Name"
COL_EXCHANGE = "\u4ea4\u6613\u6240Exchange"
COL_WEIGHT = "\u6743\u91cd(%)weight"


def normalize_code(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    digits = re.sub(r"\D", "", text)
    return digits.zfill(6) if digits else ""


def load_official(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=0, dtype=object)
    required = {
        COL_DATE,
        COL_INDEX_CODE,
        COL_INDEX_NAME,
        COL_STOCK_CODE,
        COL_STOCK_NAME,
        COL_EXCHANGE,
        COL_WEIGHT,
    }
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"Official workbook is missing columns: {missing}")
    frame = raw.rename(
        columns={
            COL_DATE: "snapshot_date",
            COL_INDEX_CODE: "index_code",
            COL_INDEX_NAME: "index_name",
            COL_STOCK_CODE: "stock_code",
            COL_STOCK_NAME: "stock_name",
            COL_EXCHANGE: "exchange",
            COL_WEIGHT: "weight_pct",
        }
    )[[
        "snapshot_date",
        "index_code",
        "index_name",
        "stock_code",
        "stock_name",
        "exchange",
        "weight_pct",
    ]].copy()
    frame["stock_code"] = frame["stock_code"].map(normalize_code)
    frame["index_code"] = frame["index_code"].map(normalize_code)
    frame["snapshot_date"] = pd.to_datetime(
        frame["snapshot_date"].astype(str), format="%Y%m%d", errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    frame["weight_pct"] = pd.to_numeric(frame["weight_pct"], errors="coerce")
    frame = frame[frame["stock_code"].str.fullmatch(r"\d{6}", na=False)].copy()
    return frame.sort_values("stock_code").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latest", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--official-xlsx", type=Path, required=True)
    parser.add_argument("--data-output-dir", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()
    args.data_output_dir.mkdir(parents=True, exist_ok=True)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)

    latest = pd.read_csv(args.latest, dtype={"stock_code": str, "index_code": str})
    history = pd.read_csv(args.history, dtype={"stock_code": str, "index_code": str})
    official = load_official(args.official_xlsx)
    official_path = args.data_output_dir / "official_csi300_closeweight_20260731.csv"
    official.to_csv(official_path, index=False, encoding="utf-8-sig")

    occurrences = latest.groupby("stock_code", as_index=True).size().rename("sina_occurrences")
    duplicate_codes = occurrences[occurrences.gt(1)].index
    duplicates = latest[latest["stock_code"].isin(duplicate_codes)].copy()
    duplicates["occurrences"] = duplicates["stock_code"].map(occurrences)
    duplicates.to_csv(
        args.data_output_dir / "sina_csi300_latest_duplicate_rows.csv",
        index=False,
        encoding="utf-8-sig",
    )

    official_codes = set(official["stock_code"])
    latest_codes = set(latest["stock_code"])
    missing_from_sina = sorted(official_codes - latest_codes)
    extra_in_sina = sorted(latest_codes - official_codes)
    comparison = official.copy()
    comparison["sina_occurrences"] = comparison["stock_code"].map(occurrences).fillna(0).astype(int)
    comparison["in_sina_latest"] = comparison["sina_occurrences"].gt(0)
    comparison["comparison_status"] = comparison["sina_occurrences"].map(
        lambda value: "missing_in_sina" if value == 0 else ("duplicated_in_sina" if value > 1 else "matched_once")
    )
    comparison.to_csv(
        args.data_output_dir / "csi300_official_vs_sina_latest.csv",
        index=False,
        encoding="utf-8-sig",
    )

    history_open = history[history["exclusion_date"].isna() | history["exclusion_date"].eq("")].copy()
    history_open_codes = set(history_open["stock_code"])
    stale_open_codes = sorted(history_open_codes - official_codes)
    history_open["in_official_20260731"] = history_open["stock_code"].isin(official_codes)
    history_open.to_csv(
        args.data_output_dir / "sina_csi300_history_open_ended_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    audit = {
        "official_snapshot": {
            "source_file": str(args.official_xlsx),
            "snapshot_dates": sorted(official["snapshot_date"].dropna().unique().tolist()),
            "rows": int(len(official)),
            "unique_codes": int(official["stock_code"].nunique()),
            "duplicate_rows": int(official.duplicated("stock_code", keep=False).sum()),
            "weight_sum_pct": float(official["weight_pct"].sum()),
            "missing_weights": int(official["weight_pct"].isna().sum()),
            "index_codes": sorted(official["index_code"].dropna().unique().tolist()),
        },
        "sina_latest": {
            "rows": int(len(latest)),
            "unique_codes": int(latest["stock_code"].nunique()),
            "duplicate_code_count": int(len(duplicate_codes)),
            "duplicated_rows": int(len(duplicates)),
            "duplicate_codes": sorted(duplicate_codes.tolist()),
            "missing_official_code_count": len(missing_from_sina),
            "missing_official_codes": missing_from_sina,
            "extra_code_count": len(extra_in_sina),
            "extra_codes": extra_in_sina,
            "exact_set_match_official": not missing_from_sina and not extra_in_sina,
        },
        "sina_history": {
            "rows": int(len(history)),
            "unique_codes": int(history["stock_code"].nunique()),
            "max_inclusion_date": str(history["inclusion_date"].dropna().max()),
            "max_exclusion_date": str(history["exclusion_date"].dropna().max()),
            "open_ended_rows": int(len(history_open)),
            "open_ended_codes_not_in_official_20260731_count": len(stale_open_codes),
            "open_ended_codes_not_in_official_20260731": stale_open_codes,
        },
        "recommended_anchor": {
            "path": str(official_path),
            "reason": "Official 300-row close-weight snapshot; use Sina only for cross-checking.",
        },
    }
    args.audit_output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

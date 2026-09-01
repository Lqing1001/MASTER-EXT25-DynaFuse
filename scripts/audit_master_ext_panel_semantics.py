"""Audit membership-panel counts and effective modeling samples in MASTER-EXT25."""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np


PARTITIONS = {
    "training": (20100104, 20191224),
    "validation": (20200102, 20211224),
    "test": (20220104, 20251231),
}


def package_counts(strict_root: Path, universe: str) -> dict:
    base = strict_root / "master_input" / universe / "by_year"
    totals = {
        name: Counter(
            membership_rows=0,
            finite_label_rows=0,
            zero_stock_feature_rows=0,
            zero_stock_feature_finite_label_rows=0,
            post_tail_trim_training_rows=0,
        )
        for name in PARTITIONS
    }
    daily_finite: dict[str, Counter[int]] = {name: Counter() for name in PARTITIONS}

    for year_dir in sorted(base.glob("year=*")):
        features = np.load(year_dir / "features.npy", mmap_mode="r")
        labels = np.load(year_dir / "labels.npy", mmap_mode="r")
        dates = np.load(year_dir / "dates.npy", mmap_mode="r")
        for name, (start, end) in PARTITIONS.items():
            selected = (dates >= start) & (dates <= end)
            if not selected.any():
                continue
            part_dates = np.asarray(dates[selected])
            part_labels = np.asarray(labels[selected])
            finite = np.isfinite(part_labels)
            stock_features = np.asarray(features[selected, :158])
            zero_stock = np.all(stock_features == 0.0, axis=1)
            totals[name]["membership_rows"] += int(selected.sum())
            totals[name]["finite_label_rows"] += int(finite.sum())
            totals[name]["zero_stock_feature_rows"] += int(zero_stock.sum())
            totals[name]["zero_stock_feature_finite_label_rows"] += int((zero_stock & finite).sum())
            unique_dates, finite_counts = np.unique(part_dates[finite], return_counts=True)
            daily_finite[name].update(
                {int(day): int(count) for day, count in zip(unique_dates, finite_counts)}
            )

    for name in PARTITIONS:
        if name == "training":
            totals[name]["post_tail_trim_training_rows"] = sum(
                count - 2 * int(0.025 * count) for count in daily_finite[name].values()
            )
        totals[name]["eligible_cross_sections"] = len(daily_finite[name])
    return {name: dict(values) for name, values in totals.items()}


def tradability_counts(database: Path, strict_root: Path, universe: str) -> dict:
    index_code = {"csi300": "000300", "csi800": "000906"}[universe]
    base = strict_root / "master_input" / universe / "by_year"
    result = {
        name: Counter(membership_rows=0, current_untradable_rows=0, current_untradable_finite_label_rows=0)
        for name in PARTITIONS
    }
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        for year_dir in sorted(base.glob("year=*")):
            year = int(year_dir.name.split("=")[1])
            rows = connection.execute(
                """
                SELECT CAST(REPLACE(m.trade_date, '-', '') AS INTEGER),
                       CASE WHEN SUBSTR(m.symbol, 1, 1) = '6'
                            THEN 'SH' || m.symbol ELSE 'SZ' || m.symbol END AS instrument,
                       CASE WHEN q.symbol IS NOT NULL AND q.volume > 0 AND q.amount > 0
                            THEN 1 ELSE 0 END AS tradable
                FROM membership AS m
                LEFT JOIN quotes AS q
                  ON q.trade_date = m.trade_date AND q.symbol = m.symbol
                WHERE m.index_code = ? AND m.trade_date >= ? AND m.trade_date <= ?
                ORDER BY m.trade_date, instrument
                """,
                (index_code, f"{year}-01-01", f"{year}-12-31"),
            ).fetchall()
            dates = np.load(year_dir / "dates.npy", mmap_mode="r")
            instruments = np.load(year_dir / "instruments.npy", mmap_mode="r")
            labels = np.load(year_dir / "labels.npy", mmap_mode="r")
            sql_dates = np.array([row[0] for row in rows], dtype=np.int32)
            sql_instruments = np.array([row[1].encode("ascii") for row in rows], dtype="S8")
            tradable = np.array([row[2] for row in rows], dtype=bool)
            if not (
                len(rows) == len(dates)
                and np.array_equal(sql_dates, np.asarray(dates))
                and np.array_equal(sql_instruments, np.asarray(instruments))
            ):
                raise RuntimeError(
                    f"SQLite/package key mismatch for {universe} {year}: "
                    f"sql_rows={len(rows)}, package_rows={len(dates)}, "
                    f"sql_first={rows[:3]}, "
                    f"package_first={list(zip(np.asarray(dates[:3]).tolist(), np.asarray(instruments[:3]).astype('U').tolist()))}"
                )
            for name, (start, end) in PARTITIONS.items():
                selected = (dates >= start) & (dates <= end)
                if not selected.any():
                    continue
                finite = np.isfinite(np.asarray(labels[selected]))
                current_untradable = ~tradable[selected]
                result[name]["membership_rows"] += int(selected.sum())
                result[name]["current_untradable_rows"] += int(current_untradable.sum())
                result[name]["current_untradable_finite_label_rows"] += int(
                    (current_untradable & finite).sum()
                )
    finally:
        connection.close()
    return {name: dict(values) for name, values in result.items()}


def membership_overlap(source_root: Path) -> dict:
    base = source_root / "master_input" / "by_year"
    daily_extra: Counter[int] = Counter()
    csi800_daily_size: dict[int, int] = {}
    totals = Counter()
    by_year: dict[str, dict[str, int]] = {}
    calendar: set[int] = set()

    for year_dir in sorted(base.glob("year=*")):
        dates = np.load(year_dir / "dates.npy", mmap_mode="r")
        csi300 = np.load(year_dir / "is_csi300.npy", mmap_mode="r")
        csi800 = np.load(year_dir / "is_csi800.npy", mmap_mode="r")
        csi300_only = np.asarray(csi300) & ~np.asarray(csi800)
        year = year_dir.name.split("=")[1]
        by_year[year] = {
            "union_rows": int(len(dates)),
            "csi300_rows": int(csi300.sum()),
            "csi800_rows": int(csi800.sum()),
            "csi300_only_rows": int(csi300_only.sum()),
        }
        totals["union_rows"] += len(dates)
        totals["csi300_rows"] += int(csi300.sum())
        totals["csi800_rows"] += int(csi800.sum())
        totals["csi300_only_rows"] += int(csi300_only.sum())
        calendar.update(map(int, np.unique(dates)))
        daily_extra.update(map(int, np.asarray(dates[csi300_only])))
        unique_days, counts = np.unique(np.asarray(dates[csi800]), return_counts=True)
        csi800_daily_size.update(
            {int(day): int(count) for day, count in zip(unique_days, counts)}
        )

    ordered_calendar = sorted(calendar)
    calendar_pos = {day: pos for pos, day in enumerate(ordered_calendar)}
    extra_dates = sorted(daily_extra, key=calendar_pos.get)
    blocks: list[dict[str, int]] = []
    if extra_dates:
        start = previous = extra_dates[0]
        for day in extra_dates[1:]:
            if calendar_pos[day] != calendar_pos[previous] + 1:
                blocks.append(
                    {
                        "start": start,
                        "end": previous,
                        "trading_days": calendar_pos[previous] - calendar_pos[start] + 1,
                        "rows": sum(
                            daily_extra[value]
                            for value in extra_dates
                            if calendar_pos[start] <= calendar_pos[value] <= calendar_pos[previous]
                        ),
                    }
                )
                start = day
            previous = day
        blocks.append(
            {
                "start": start,
                "end": previous,
                "trading_days": calendar_pos[previous] - calendar_pos[start] + 1,
                "rows": sum(
                    daily_extra[value]
                    for value in extra_dates
                    if calendar_pos[start] <= calendar_pos[value] <= calendar_pos[previous]
                ),
            }
        )

    return {
        "totals": dict(totals),
        "identity_check": {
            "union_minus_csi800": totals["union_rows"] - totals["csi800_rows"],
            "csi300_only_rows": totals["csi300_only_rows"],
            "equal": totals["union_rows"] - totals["csi800_rows"] == totals["csi300_only_rows"],
        },
        "csi800_daily_size_distribution": dict(sorted(Counter(csi800_daily_size.values()).items())),
        "csi800_799_member_dates": sorted(day for day, count in csi800_daily_size.items() if count == 799),
        "csi300_only_active_days": len(daily_extra),
        "csi300_only_rows_per_day_distribution": dict(sorted(Counter(daily_extra.values()).items())),
        "csi300_only_blocks": blocks,
        "by_year": by_year,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--strict-root", type=Path, required=True)
    parser.add_argument("--sqlite-db", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = {
        "definitions": {
            "membership_rows": "date-security rows selected by the point-in-time membership mask",
            "finite_label_rows": "rows eligible for IC/RankIC after finite prediction-label masking",
            "post_tail_trim_training_rows": "finite-label training rows after dropping 2.5% from each daily label tail",
            "zero_stock_feature_rows": "membership rows whose 158 normalized stock features are all zero; this is a missing-feature proxy, not an exchange tradability flag",
        },
        "effective_samples": {
            universe: {
                "package": package_counts(args.strict_root, universe),
                "current_tradability": tradability_counts(args.sqlite_db, args.strict_root, universe),
            }
            for universe in ("csi300", "csi800")
        },
        "membership_overlap": membership_overlap(args.source_root),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

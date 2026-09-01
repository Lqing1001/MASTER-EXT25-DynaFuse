"""Merge full CSI 800 and output validations into the final reconstruction manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reconstruction-audit", type=Path, required=True)
    parser.add_argument("--csi800-validation", type=Path, required=True)
    parser.add_argument("--output-validation", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    args = parser.parse_args()

    audit = json.loads(args.reconstruction_audit.read_text(encoding="utf-8"))
    csi800 = json.loads(args.csi800_validation.read_text(encoding="utf-8"))
    validation = json.loads(args.output_validation.read_text(encoding="utf-8"))
    validation.get("files", {}).pop("manifest.json", None)
    args.output_validation.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    partial = audit.pop("csi800_validation", None)
    audit["status"] = validation["status"]
    audit["csi800_validation"] = {
        "scope": "all nine CSMAR IDX_Smprat workbooks",
        "validation_file": str(args.csi800_validation.resolve()),
        "workbook_count": csi800["workbook_count"],
        "comparisons": csi800["comparisons"],
    }
    audit["output_validation"] = {
        "validation_file": str(args.output_validation.resolve()),
        "status": validation["status"],
        "checks": validation["checks"],
        "metrics": validation["metrics"],
        "file_hashes": validation["files"],
    }
    audit["superseded_partial_csi800_scan"] = partial
    audit["scope_notes"] = [
        "The requested 2010-2025 range begins on the first available 2010 trading day.",
        "Intervals active on 2010-01-04 are left-censored at the requested boundary.",
        "Intervals active on 2025-12-31 are right-censored at the requested boundary.",
        "The official 2026-07-31 close-weight file is the reconstruction anchor.",
    ]
    payload = json.dumps(audit, ensure_ascii=False, indent=2)
    args.reconstruction_audit.write_text(payload, encoding="utf-8")
    args.dataset_manifest.write_text(payload, encoding="utf-8")
    print(json.dumps({
        "status": audit["status"],
        "csi800": audit["csi800_validation"]["comparisons"],
        "checks": audit["output_validation"]["checks"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

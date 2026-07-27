from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FREEZE_ID = "OATS-PUBLISHED-BASELINE-20260727-R1"
PARENT_ID = "OATS-TRUSTFIX-PRE-PUBLISHED-BASELINE-20260727-R1"
OUTPUT = ROOT / "docs" / "freezes" / FREEZE_ID


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def collect() -> list[Path]:
    explicit = [
        ROOT / "src" / "oats_external" / "adapters" / "oasis_tsc.py",
        ROOT / "src" / "oats_external" / "adapters" / "__init__.py",
        ROOT / "src" / "oats_external" / "types.py",
        ROOT / "src" / "oats_external" / "realcal_bridge.py",
        ROOT / "scripts" / "prepare_oasis_arrival_forecast.py",
        ROOT / "scripts" / "run_oasis_realcal.py",
        ROOT / "scripts" / "analyze_oasis_published_baseline.py",
        ROOT / "scripts" / "verify_oasis_published_baseline.py",
        ROOT / "scripts" / "freeze_oasis_published_baseline.py",
        ROOT / "tests" / "external_baselines" / "test_oasis_tsc.py",
        ROOT / "tests" / "external_baselines" / "test_adapter_isolation.py",
        ROOT / "tests" / "external_baselines" / "test_method_disposition.py",
        ROOT / "tests" / "external_baselines" / "test_realcal_bridge.py",
        ROOT
        / "docs"
        / "freezes"
        / PARENT_ID
        / "FREEZE_MANIFEST.json",
        ROOT
        / "docs"
        / "freezes"
        / PARENT_ID
        / "FREEZE_VERIFICATION_RECEIPT.json",
    ]
    paper = ROOT / "paper" / "revision" / "format_unified_2026-07-29"
    explicit.extend(
        paper / name
        for name in (
            "main2026-7-29.tex",
            "main2026-7-29.pdf",
            "main2026-7-29.bbl",
            "main2026-7-29.log",
            "supplement2026-7-29.tex",
            "supplement2026-7-29.pdf",
            "supplement2026-7-29.log",
            "reference.bib",
        )
    )
    recursive: list[Path] = []
    recursive.extend(
        path
        for path in (
            ROOT / "reports" / "published_baseline_oasis_20260727"
        ).rglob("*")
        if path.is_file() and "papers" not in path.parts
    )
    recursive.extend(
        path
        for path in (
            ROOT / "results" / "published_baseline_oasis_20260727_r1"
        ).rglob("*")
        if path.is_file()
    )
    paths = sorted(
        set(explicit + recursive),
        key=lambda path: path.relative_to(ROOT).as_posix(),
    )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing freeze input: {missing}")
    return paths


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit(f"refusing to overwrite freeze: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    paths = collect()
    records = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    ]
    manifest: dict[str, object] = {
        "freeze_id": FREEZE_ID,
        "parent_freeze_id": PARENT_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Published Oasis baseline preregistration, source, validation/formal "
            "results, paired analysis, provenance, tests, and synchronized "
            "manuscript PDFs. The copyrighted research PDF and figure-rendering "
            "code are excluded."
        ),
        "formal_result_root": "results/published_baseline_oasis_20260727_r1/formal",
        "formal_cell_count": 50,
        "paper_doi": "10.1109/TSC.2024.3354240",
        "file_count": len(records),
        "files": records,
    }
    manifest["manifest_hash"] = canonical_hash(manifest)
    manifest_path = OUTPUT / "FREEZE_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    archive = OUTPUT / f"{FREEZE_ID}.zip"
    with zipfile.ZipFile(
        archive,
        mode="x",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        allowZip64=True,
    ) as handle:
        handle.write(manifest_path, "FREEZE_MANIFEST.json")
        for path in paths:
            handle.write(path, path.relative_to(ROOT).as_posix())

    verified = 0
    mismatches: list[str] = []
    expected = {record["path"]: record for record in records}
    with zipfile.ZipFile(archive, mode="r") as handle:
        names = handle.namelist()
        expected_names = {"FREEZE_MANIFEST.json", *expected}
        unexpected = sorted(set(names).difference(expected_names))
        missing = sorted(expected_names.difference(names))
        archived_manifest = json.loads(
            handle.read("FREEZE_MANIFEST.json").decode("utf-8")
        )
        if archived_manifest["manifest_hash"] != manifest["manifest_hash"]:
            mismatches.append("embedded_manifest_hash")
        for name, record in expected.items():
            data = handle.read(name)
            if (
                len(data) != int(record["size_bytes"])
                or sha256_bytes(data) != record["sha256"]
            ):
                mismatches.append(name)
            else:
                verified += 1
    receipt: dict[str, object] = {
        "status": (
            "PASS"
            if not mismatches and not missing and not unexpected
            else "FAIL"
        ),
        "freeze_id": FREEZE_ID,
        "parent_freeze_id": PARENT_ID,
        "manifest_hash": manifest["manifest_hash"],
        "manifest_sha256": sha256_file(manifest_path),
        "archive_path": archive.relative_to(ROOT).as_posix(),
        "archive_size_bytes": archive.stat().st_size,
        "archive_sha256": sha256_file(archive),
        "expected_file_count": len(records),
        "verified_file_count": verified,
        "missing_archive_entries": missing,
        "unexpected_archive_entries": unexpected,
        "hash_or_size_mismatches": mismatches,
        "copyrighted_source_pdf_included": False,
        "figure_rendering_code_included": False,
    }
    receipt["receipt_hash"] = canonical_hash(receipt)
    receipt_path = OUTPUT / "FREEZE_VERIFICATION_RECEIPT.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, ensure_ascii=False))
    if receipt["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""TraceGuard dataset curation pipeline.

Builds ~/data/curated/{images/, manifest.csv, audit_report.md} from the raw
datasets under ~/data, with deterministic (seed-42) sampling, collision-safe
hashed filenames, Pillow verification, and sha256 dedupe.

Usage: python3 -m traceguard_data.run --stage all
"""

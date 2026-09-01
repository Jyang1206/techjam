"""CLI entry point.

    python3 -m traceguard_data.run --stage df40_fakes
    python3 -m traceguard_data.run --stage all

Each stage prints STAGE_<name>_OK / STAGE_<name>_FAIL markers for log
monitoring, guards free disk before starting, and writes its part manifest
atomically, so stages are safe to rerun individually.
"""
import argparse
import sys
import traceback

from . import config, util
from .df40 import stage_df40_fakes, stage_df40_reals, stage_df40_unseen
from .finalize import stage_finalize
from .scenes import (stage_commfor, stage_genimagepp, stage_sid_set,
                     stage_wildfake)

STAGE_FUNCS = {
    "df40_fakes": stage_df40_fakes,
    "df40_unseen": stage_df40_unseen,
    "df40_reals": stage_df40_reals,
    "wildfake": stage_wildfake,
    "sid_set": stage_sid_set,
    "commfor": stage_commfor,
    "genimagepp": stage_genimagepp,
    "finalize": stage_finalize,
}


def run_stage(name: str) -> bool:
    try:
        util.check_disk(name)
        STAGE_FUNCS[name]()
        print(f"STAGE_{name}_OK", flush=True)
        return True
    except Exception:
        traceback.print_exc()
        print(f"STAGE_{name}_FAIL", flush=True)
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=[*STAGE_FUNCS, "all"])
    args = ap.parse_args()
    if args.stage != "all":
        sys.exit(0 if run_stage(args.stage) else 1)
    ok = True
    for name in config.STAGES:  # sources sequentially, then finalize
        ok = run_stage(name) and ok
    if ok:
        ok = run_stage("finalize")
    else:
        print("SKIPPING finalize: a source stage failed", flush=True)
    print("ALL_STAGES_DONE" if ok else "ALL_STAGES_FAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

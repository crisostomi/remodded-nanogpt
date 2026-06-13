"""CLI: upload an existing run's manifest+artifacts to a tracking backend.

    python -m nano.runtime.flywheel --manifest experiments/runs/<run_id>/manifest.json

Flywheel itself is a stub until creds/docs are available; ``--backend`` lets you
target the no-op or local-jsonl backends in the meantime.
"""

from __future__ import annotations

import argparse
import sys

from nano.runtime.tracking import upload_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Upload a run to a tracking backend.")
    parser.add_argument("--manifest", required=True, help="Path to a run manifest.json")
    parser.add_argument(
        "--backend",
        default=None,
        help="Override backend (noop|local|flywheel); defaults to manifest tracking.backend",
    )
    args = parser.parse_args(argv)

    result = upload_run(args.manifest, backend_name=args.backend)
    if result["ok"]:
        print(f"Upload OK via {result['backend']}")
        return 0
    print(f"Upload FAILED via {result['backend']}: {result['error']}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

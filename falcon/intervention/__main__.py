"""Command-line interface for stage interventions (Task T5, Plan Appendix B).

Example:
    python -m falcon.intervention --runs-root runs --target-run fail_001 \
        --source-run ref_001 --round 1 --stage compression --mode restore \
        [--client-ids a,b] [--json out.json]

    # windowed intervention (T13, Plan §13.5), inclusive rounds t1..t2:
    python -m falcon.intervention --runs-root runs --target-run fail_001 \
        --source-run ref_001 --round-window 2:9 --stage compression \
        --mode restore
"""

from __future__ import annotations

import argparse
from pathlib import Path

from falcon.schema import STAGES, InterventionSpecification

from .engine import apply_intervention

_MODES = ("restore", "inject", "sham")


def _parse_round_window(text: str) -> tuple[int, int]:
    """Parse ``t1:t2`` into an inclusive ``(t1, t2)`` round window."""
    try:
        start, end = text.split(":")
        window = (int(start), int(end))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--round-window must be T1:T2 with integer rounds, got {text!r}"
        ) from None
    return window


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Apply a FALCON stage intervention and replay the run"
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        required=True,
        help="recorder root directory (runs live under <runs-root>/runs/<run-id>)",
    )
    parser.add_argument("--target-run", required=True, help="run to re-execute")
    parser.add_argument("--source-run", required=True, help="run supplying the replacement state")
    round_group = parser.add_mutually_exclusive_group(required=True)
    round_group.add_argument(
        "--round", type=int, dest="round_id",
        help="single intervention round",
    )
    round_group.add_argument(
        "--round-window", type=_parse_round_window, dest="round_window",
        metavar="T1:T2",
        help="inclusive intervention round window (T13); replaces --round",
    )
    parser.add_argument("--stage", required=True, choices=list(STAGES))
    parser.add_argument("--mode", required=True, choices=list(_MODES))
    parser.add_argument(
        "--client-ids",
        default=None,
        help="comma-separated client ids (local/compression stages only)",
    )
    parser.add_argument("--json", type=Path, dest="json_path", help="write the InterventionResult here")
    args = parser.parse_args(argv)

    scope = {}
    if args.client_ids:
        scope["client_ids"] = [
            client_id.strip()
            for client_id in args.client_ids.split(",")
            if client_id.strip()
        ]
    spec = InterventionSpecification(
        target_run_id=args.target_run,
        source_run_id=args.source_run,
        # schema keeps round_id required; it is ignored when round_window is set
        round_id=args.round_id if args.round_window is None else args.round_window[0],
        round_window=args.round_window,
        stage=args.stage,
        mode=args.mode,
        scope=scope,
    )
    result = apply_intervention(spec, args.runs_root)

    print(f"Valid: {result.valid}")
    if result.reason is not None:
        print(f"Reason: {result.reason}")
    if result.outcome_metrics:
        print("Outcome metrics:")
        width = max(map(len, result.outcome_metrics))
        for name, value in result.outcome_metrics.items():
            print(f"  {name:<{width}}  {value:.6f}")

    if args.json_path is not None:
        args.json_path.write_text(
            result.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
    return 0 if result.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())

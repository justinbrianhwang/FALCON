"""Command-line interface for end-to-end FALCON attribution."""

from __future__ import annotations

import argparse
from pathlib import Path

from .analyze import analyze_pair, load_ground_truth
from .report import render_markdown


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze a matched FALCON run pair")
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--failure", required=True)
    parser.add_argument("--metric", required=True)
    direction = parser.add_mutually_exclusive_group()
    direction.add_argument(
        "--higher-is-better", dest="higher_is_better", action="store_true"
    )
    direction.add_argument(
        "--lower-is-better", dest="higher_is_better", action="store_false"
    )
    parser.set_defaults(higher_is_better=True)
    parser.add_argument("--min-gap", type=float, default=0.005)
    parser.add_argument("--sham-tolerance", type=float, default=1e-9)
    parser.add_argument("--decisive-margin", type=float, default=0.05)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", type=Path, dest="json_path")
    args = parser.parse_args(argv)

    report, interventions = analyze_pair(
        args.runs_root,
        args.reference,
        args.failure,
        metric=args.metric,
        higher_is_better=args.higher_is_better,
        min_gap=args.min_gap,
        sham_tolerance=args.sham_tolerance,
        decisive_margin=args.decisive_margin,
    )
    ground_truth = (
        load_ground_truth(args.runs_root, args.failure)
        if report.pair.status != "INVALID_PAIR"
        else None
    )
    markdown = render_markdown(report, interventions, ground_truth=ground_truth)

    if args.output is None:
        print(markdown, end="")
    else:
        args.output.write_text(markdown, encoding="utf-8")
    if args.json_path is not None:
        args.json_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return 0 if report.pair.status != "INVALID_PAIR" else 1


if __name__ == "__main__":
    raise SystemExit(main())

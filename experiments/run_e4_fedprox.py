"""E4 - FedProx algorithm-generality localization matrix (T26).

    python experiments/run_e4_fedprox.py
    python experiments/run_e4_fedprox.py --smoke

Uses the MNIST reference configuration with FedProx local training. SCAFFOLD
is out of scope because its server control variates require recorder-visible
server state. Writes summary.json and e4_table.md under results/.
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from run_coauthor_cifar import FAILURES, _record  # noqa: E402

from falcon.reporting.analyze import analyze_pair_vector  # noqa: E402
from falcon.reporting.report import render_markdown  # noqa: E402
from falcon.schema import RunConfig  # noqa: E402


def _mu_id(mu: float) -> str:
    return str(mu).replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke", action="store_true", help="FedProx mu=0.1, selection only, 4 rounds"
    )
    args = parser.parse_args()

    base = yaml.safe_load(
        (REPO / "configs" / "cases" / "mnist_reference.yaml").read_text(
            encoding="utf-8"
        )
    )
    base["local"] = {**base["local"], "algorithm": "fedprox"}
    base["rounds"] = 4 if args.smoke else 5
    mus = (0.1,) if args.smoke else (0.1, 0.01)
    failures = FAILURES[:1] if args.smoke else FAILURES
    active_rounds = (1, 3) if args.smoke else (1, 4)
    out = REPO / "results" / ("e4_fedprox_smoke" if args.smoke else "e4_fedprox")
    out.mkdir(parents=True, exist_ok=True)

    summary = []
    for mu in mus:
        mu_id = _mu_id(mu)
        cfg_dict = {**base, "local": {**base["local"], "prox_mu": mu}}
        ref_id = f"ref_mu_{mu_id}"
        ref_cfg = RunConfig(**{**cfg_dict, "run_id": ref_id, "failure": None})
        print(f"[e4] reference: mu={mu} ({ref_cfg.rounds} rounds)...", flush=True)
        _record(out, ref_cfg)

        for name, original_spec in failures:
            spec = original_spec.model_copy(update={"active_rounds": active_rounds})
            fail_id = f"fail_mu_{mu_id}_{name}"
            fail_cfg = RunConfig(**{**cfg_dict, "run_id": fail_id, "failure": spec})
            print(f"[e4] failure: mu={mu}, stage={name}...", flush=True)
            _record(out, fail_cfg)

            metrics = {"accuracy": {"higher_is_better": True}}
            if name in {"selection", "aggregation"}:
                metrics["class_5_accuracy"] = {"higher_is_better": True}
            print(
                f"[e4] attribution: mu={mu}, stage={name}, "
                f"metrics={','.join(metrics)}...",
                flush=True,
            )
            try:
                reports, interventions = analyze_pair_vector(
                    out,
                    ref_id,
                    fail_id,
                    metrics=metrics,
                    min_gap=0.005,
                    sham_tolerance=1e-9,
                )
                for metric, report in reports.items():
                    report_path = out / f"report_mu_{mu_id}_{name}_{metric}.md"
                    report_path.write_text(
                        render_markdown(report, interventions, ground_truth=spec),
                        encoding="utf-8",
                    )
                    row = {
                        "prox_mu": mu,
                        "failure": name,
                        "ground_truth": spec.stage,
                        "metric": metric,
                        "outcome": report.outcome,
                        "prediction": (
                            report.origin_ranking[0] if report.origin_ranking else None
                        ),
                        "origin_set": report.origin_set,
                        "gap": report.failure_gap,
                        "notes": report.notes,
                    }
                    summary.append(row)
                    print(f"  -> {ascii(row)}", flush=True)
            except Exception as error:
                for metric in metrics:
                    row = {
                        "prox_mu": mu,
                        "failure": name,
                        "ground_truth": spec.stage,
                        "metric": metric,
                        "error": f"{type(error).__name__}: {error}",
                    }
                    summary.append(row)
                    print(f"  -> {ascii(row)}", flush=True)

    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    lines = [
        "# E4 FedProx localization",
        "",
        "| prox_mu | failure | metric | outcome | prediction | gap |",
        "|---:|---|---|---|---|---:|",
    ]
    for row in summary:
        if "error" in row:
            cells = ["ERROR", "-", row["error"]]
        else:
            gap = row["gap"].get(row["metric"])
            cells = [
                str(row["outcome"]),
                str(row["prediction"]),
                "n/a" if gap is None else f"{gap:+.4f}",
            ]
        lines.append(
            f"| {row['prox_mu']} | {row['failure']} | {row['metric']} | "
            + " | ".join(cells)
            + " |"
        )
    (out / "e4_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[e4] wrote {out / 'summary.json'} and {out / 'e4_table.md'}", flush=True)


if __name__ == "__main__":
    main()

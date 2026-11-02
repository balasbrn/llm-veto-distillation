"""Command line: ``vd demo``, ``vd import``, ``vd run``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from veto_distill import cmf_import, data, evaluate, report, synthetic


def _examples(cards, n: int = 3):
    take = [c for c in cards if c.teacher_take]
    skip = [c for c in cards if not c.teacher_take]
    picks = take[:: max(1, len(take) // 2)][:2] + skip[:1]
    return picks[:n]


def cmd_demo(args) -> int:
    cards = synthetic.make_dataset(args.n, args.seed)
    if args.save_data:
        data.save(cards, Path(args.save_data))
    res = evaluate.run(cards, args.train_share)
    path = report.write(res, Path(args.out), "Distillation run — SYNTHETIC demo data",
                        "synthetic cards and a hand-written heuristic teacher "
                        "(`veto_distill.synthetic`); no market data, no LLM. Numbers "
                        "here only show the pipeline works.", _examples(cards))
    print(f"wrote {path}")
    return 0


def cmd_import(args) -> int:
    root = Path(args.data_root).expanduser() if args.data_root else None
    all_cards, inputs = [], []
    for spec in args.source:
        try:
            group, pair, period = spec.split(":")
        except ValueError:
            print(f"--source must be GROUP:PAIR:PERIOD, not {spec}", file=sys.stderr)
            return 2
        if root is None:
            print("--data-root is required", file=sys.stderr)
            return 2
        cards_path, teacher_path = cmf_import.default_paths(root, pair, period, args.run)
        cards, counts = cmf_import.import_files(cards_path, teacher_path, group)
        print(f"{group}: {counts}")
        all_cards += cards
        inputs += [{"group": group, "file": p.name, "sha256": data.sha256(p)}
                   for p in (cards_path, teacher_path)]
    out = Path(args.out)
    data.save(all_cards, out)
    manifest = {"dataset": out.name, "dataset_sha256": data.sha256(out),
                "cards": len(all_cards), "teacher_run": args.run, "inputs": inputs}
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(all_cards)} cards to {args.out}")
    return 0


def cmd_sample(args) -> int:
    cards = data.public_sample(data.load(Path(args.data)), args.n)
    out = Path(args.out)
    data.save(cards, out)
    print(f"wrote {len(cards)} anonymous cards to {out} (sha256 {data.sha256(out)})")
    return 0


def cmd_run(args) -> int:
    cards = data.load(Path(args.data))
    res = evaluate.run(cards, args.train_share)
    res["dataset_sha256"] = data.sha256(Path(args.data))
    path = report.write(res, Path(args.out), args.title, args.source_note, _examples(cards))
    print(f"wrote {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="vd", description="Distil an LLM trade veto into small models.")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="run the whole pipeline on synthetic data")
    d.add_argument("--n", type=int, default=1200)
    d.add_argument("--seed", type=int, default=7)
    d.add_argument("--train-share", type=float, default=0.6)
    d.add_argument("--out", default="results/synthetic")
    d.add_argument("--save-data", help="also write the synthetic dataset (JSONL)")
    d.set_defaults(fn=cmd_demo)

    i = sub.add_parser("import", help="join upstream cards and teacher answers into a dataset")
    i.add_argument("--data-root", required=True, help="upstream data folder")
    i.add_argument("--source", action="append", required=True,
                   help="GROUP:PAIR:PERIOD, e.g. ADA:ADAUSD:dev (repeatable)")
    i.add_argument("--run", default="gemini-2.5-flash-v2-think0",
                   help="teacher run name in the upstream file name")
    i.add_argument("--out", default="data/teacher_labels.jsonl")
    i.set_defaults(fn=cmd_import)

    s = sub.add_parser("sample", help="anonymous public sample of a dataset (dates removed)")
    s.add_argument("--data", default="data/teacher_labels.jsonl")
    s.add_argument("--n", type=int, default=500)
    s.add_argument("--out", default="sample/anonymous_cards.jsonl")
    s.set_defaults(fn=cmd_sample)

    r = sub.add_parser("run", help="train the students and write the report")
    r.add_argument("--data", default="data/teacher_labels.jsonl")
    r.add_argument("--train-share", type=float, default=0.6)
    r.add_argument("--out", default="results/real")
    r.add_argument("--title", default="Distillation run")
    r.add_argument("--source-note", default="teacher-labelled cards imported with `vd import`.")
    r.set_defaults(fn=cmd_run)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())

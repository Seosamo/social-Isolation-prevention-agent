from __future__ import annotations

import argparse
import subprocess
import sys

from activity_agent.common.config import Config


def main(argv=None):
    # Windows redirected consoles may default to CP949, which cannot encode
    # all characters used in evidence output. Files are UTF-8 as well.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="지역 활동 위축 분석 파이프라인")
    parser.add_argument("--config", default="config/default.json")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "preprocess", "features", "report", "run-all", "dashboard", "structural", "cohorts", "readiness", "run-isolation", "policies-refresh", "llm-check"):
        commands.add_parser(name)
    for name in ("detect", "label", "evaluate"):
        commands.add_parser(name).add_argument("--task", choices=["activity", "isolation"], default="activity")
    training = commands.add_parser("train")
    training.add_argument("--task", choices=["activity", "isolation"], default="activity")
    training.add_argument("--signal", choices=["mobility", "district_mobility", "consumption"], default="mobility")
    training.add_argument("--source", choices=["weak", "reviewed"], default="weak")
    training.add_argument("--train-end")
    training.add_argument("--validation-month")
    training.add_argument("--test-month")
    training.add_argument("--holdout-region")
    review = commands.add_parser("review")
    review.add_argument("--task", choices=["activity", "isolation"], default="activity")
    review.add_argument("feature_id")
    review.add_argument("--label", type=int, choices=[-1, 0, 1], required=True)
    review.add_argument("--reviewer", required=True)
    review.add_argument("--note", required=True)
    review.add_argument("--independent", action="store_true", help="Only for evidence independent of weak rules")
    ask = commands.add_parser("ask")
    ask.add_argument("--task", choices=["activity", "isolation"], default="activity")
    ask.add_argument("question")
    ask.add_argument("--backend", choices=["local", "vllm"])
    ask.add_argument("--signal", default=None)
    ask.add_argument("--month")
    ask.add_argument("--sex")
    ask.add_argument("--age")
    ask.add_argument("--region")
    args = parser.parse_args(argv)
    config = Config.load(args.config)
    try:
        if args.command == "llm-check":
            import json
            from activity_agent.llm.vllm import VLLMClient, VLLMSettings
            print(json.dumps(VLLMClient(VLLMSettings.load(config)).check(), ensure_ascii=False))
        if args.command == "policies-refresh":
            import json
            from activity_agent.policies.collector import collect
            result = collect(config)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["collected"] else 1
        task = getattr(args, "task", "activity")
        if args.command in ("structural", "run-isolation"):
            from activity_agent.isolation.structural import load_structural_sources
            load_structural_sources(config)
        if args.command in ("cohorts", "run-isolation"):
            from activity_agent.isolation.cohorts import build_cohorts
            build_cohorts(config)
        if args.command == "run-isolation" or (args.command == "label" and task == "isolation"):
            from activity_agent.isolation.labeling import label_cohorts
            label_cohorts(config)
        if args.command in ("readiness", "run-isolation"):
            from activity_agent.isolation.training import validate_training_readiness, train_model
            readiness = validate_training_readiness(config)
            print(f"Isolation training ready: {readiness['training_ready']}", flush=True)
            if args.command == "run-isolation":
                if readiness["training_ready"]:
                    train_model(config)
                else:
                    print("Training not run. See artifacts/isolation/readiness.json for blockers.", flush=True)
        if args.command in ("validate", "preprocess", "run-all"):
            from activity_agent.data.ingest import preprocess
            preprocess(config, validate_only=args.command == "validate")
        if args.command in ("features", "run-all"):
            from activity_agent.activity.features import build_features
            build_features(config)
        if args.command == "run-all" or (args.command == "detect" and task == "activity"):
            from activity_agent.activity.detection import detect
            detect(config)
        if args.command == "run-all" or (args.command == "label" and task == "activity"):
            from activity_agent.activity.labels import label
            label(config)
        if args.command in ("report", "run-all"):
            from activity_agent.activity.reporting import report
            report(config)
        if args.command == "train" and task == "activity":
            from activity_agent.activity.training import train
            train(config, args.signal, args.source, args.train_end or "202510", args.validation_month or "202511", args.test_month or "202512", args.holdout_region)
        if args.command == "train" and task == "isolation":
            if any((args.train_end, args.validation_month, args.test_month, args.holdout_region)):
                raise ValueError("Configure isolation splits/holdout_districts in the config, then rebuild isolation labels before training")
            from activity_agent.isolation.training import train_model
            train_model(config, args.source)
        if args.command == "detect" and task == "isolation":
            from activity_agent.isolation.training import predict_isolation
            predict_isolation(config)
        if args.command == "evaluate" and task == "activity":
            from activity_agent.activity.training import evaluate
            evaluate(config)
        if args.command == "evaluate" and task == "isolation":
            from activity_agent.isolation.training import evaluate_model
            evaluate_model(config)
        if args.command == "review" and task == "activity":
            from activity_agent.activity.labels import add_review
            add_review(config, args.feature_id, args.label, args.reviewer, args.note, args.independent)
        if args.command == "review" and task == "isolation":
            from activity_agent.isolation.labeling import review_isolation
            review_isolation(config, args.feature_id, args.label, args.reviewer, args.note, args.independent)
        if args.command == "ask" and task == "activity":
            from activity_agent.llm.vllm import create_activity_agent
            print(create_activity_agent(config, backend=args.backend).answer(args.question, signal=args.signal, month=args.month,
                  sex=args.sex, age=args.age, region=args.region))
        if args.command == "ask" and task == "isolation":
            from activity_agent.llm.vllm import create_isolation_agent
            print(create_isolation_agent(config, backend=args.backend).answer(args.question, month=args.month, sex=args.sex, age=args.age, region=args.region))
        if args.command == "dashboard":
            return subprocess.call([sys.executable, "-m", "streamlit", "run", str(config.root / "app.py"),
                                    "--server.headless=true", "--", "--config", str(config.root / args.config)])
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"Error: {error}\n")
    return 0

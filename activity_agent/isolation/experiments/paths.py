"""Model storage separated from experiment result artifacts."""
from pathlib import Path
import json
import os
import shutil

from activity_agent.common.storage import write_json, fingerprint


def model_directory(run):
    run = Path(run)
    locator = run / "config/model_directory.json"
    if locator.exists():
        return (run / json.loads(locator.read_text(encoding="utf-8"))["path"]).resolve()
    return run / "models"  # Historical runs and portable result fixtures.


def configure_model_directory(run, project_root):
    run = Path(run).resolve()
    target = Path(project_root).resolve() / "activity_agent/model/experiments" / run.name
    target.mkdir(parents=True, exist_ok=True)
    write_json(run / "config/model_directory.json", {"path": os.path.relpath(target, run),
               "naming": "experiment_id", "run_id": run.name})
    return target


def migrate_models(run, project_root):
    """Move a historical run's models, recording unchanged content hashes."""
    run, root = Path(run).resolve(), Path(project_root).resolve()
    source = model_directory(run).resolve()
    target = root / "activity_agent/model/experiments" / run.name
    if source == target:
        return target
    if not source.is_relative_to(root) or not target.is_relative_to(root):
        raise ValueError("Model migration must stay inside the project")
    files = sorted(source.glob("*"))
    entries = []
    for path in files:
        if not path.is_file():
            raise ValueError(f"Unexpected model directory: {path}")
        name = "IF_BASELINE_ORIGINAL.pkl" if path.name == "operating_original.pkl" else path.name
        if (target / name).exists():
            raise FileExistsError(target / name)
        entries.append({"original_name": path.name, "name": name, "sha256": fingerprint(path)})
    target.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        shutil.move(str(source / entry["original_name"]), str(target / entry["name"]))
        if fingerprint(target / entry["name"]) != entry["sha256"]:
            raise ValueError("Model migration checksum mismatch")
    configure_model_directory(run, root)
    write_json(target / "model_migration.json", {"files": entries, "run_id": run.name,
               "original_path": os.path.relpath(source, root)})
    return target

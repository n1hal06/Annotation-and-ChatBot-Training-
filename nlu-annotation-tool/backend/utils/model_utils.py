# backend/utils/model_utils.py
import os
import json
import random
import time
import shutil
import subprocess
from glob import glob
from typing import List

# ---------- spaCy trainer (your existing function kept) ----------
def train_spacy_model(base_dir: str) -> str:
    """
    Train a minimal spaCy NER model from annotations.json and save to models/spacy_model/model_v{ts}
    """
    try:
        import spacy
        from spacy.training import Example
    except Exception as e:
        raise RuntimeError('spaCy is required for training: ' + str(e))

    backend_dir = os.path.join(base_dir, 'models')
    spacy_dir = os.path.join(backend_dir, 'spacy_model')
    os.makedirs(spacy_dir, exist_ok=True)

    data_file = os.path.join(base_dir, 'data', 'annotations.json')
    if not os.path.exists(data_file):
        raise FileNotFoundError('annotations.json not found')

    with open(data_file, 'r', encoding='utf-8') as fh:
        annotations = json.load(fh)

    # Prepare training examples: spaCy expects list of (text, {'entities': [(start,end,label), ...]})
    training_data = []
    labels = set()
    for ann in annotations:
        text = ann.get('text', '')
        ents = ann.get('entities', [])
        spacy_ents = []
        for e in ents:
            # Expect entity dict with start, end, label
            try:
                s = int(e.get('start'))
                en = int(e.get('end'))
                lab = str(e.get('label'))
                spacy_ents.append((s, en, lab))
                labels.add(lab)
            except Exception:
                continue
        if text:
            training_data.append((text, {'entities': spacy_ents}))

    if not training_data:
        raise RuntimeError('No training data available in annotations.json')

    # Create blank English model
    import spacy
    nlp = spacy.blank('en')

    if 'ner' not in nlp.pipe_names:
        ner = nlp.add_pipe('ner')
    else:
        ner = nlp.get_pipe('ner')

    for label in labels:
        ner.add_label(label)

    # Begin training
    optimizer = nlp.begin_training()

    # convert training data to spaCy Example objects for newer API
    examples = []
    from spacy.training import Example
    for text, ann in training_data:
        doc = nlp.make_doc(text)
        examples.append(Example.from_dict(doc, ann))

    # Train for a small number of epochs
    for epoch in range(10):
        random.shuffle(examples)
        losses = {}
        for example in examples:
            nlp.update([example], sgd=optimizer, drop=0.35, losses=losses)
        print(f'[model_utils] epoch {epoch+1}/10, losses={losses}')

    # Save model
    timestamp = int(time.time())
    model_version_dir = os.path.join(spacy_dir, f'model_v{timestamp}')
    os.makedirs(model_version_dir, exist_ok=True)
    nlp.to_disk(model_version_dir)

    # write metadata
    meta = {'name': 'spacy_ner', 'version': f'v{timestamp}', 'trained_at': timestamp}
    with open(os.path.join(spacy_dir, f'meta_v{timestamp}.json'), 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, indent=2)

    return model_version_dir


# ---------- Rasa trainer + helpers ----------
def annotations_to_rasa_nlu(annotations: List[dict], rasa_project_path: str) -> str:
    """
    Convert annotations list into Rasa-style data/nlu.yml file inside rasa_project_path/data/nlu.yml
    annotations: list of {"text":..., "intent":..., "entities":[{"start":int,"end":int,"label":str}, ...]}
    Returns path to written nlu.yml
    """
    import yaml

    data_dir = os.path.join(rasa_project_path, "data")
    os.makedirs(data_dir, exist_ok=True)

    # Build examples grouped by intent (Rasa YAML format v2.0 simple)
    intents_map = {}
    for ann in annotations:
        text = ann.get("text", "").strip()
        if not text:
            continue
        intent = ann.get("intent", "unknown_intent")
        entities = ann.get("entities", [])
        if not entities:
            # plain example
            example = text
        else:
            # Create inline markup [entity_text](label) using span offsets
            spans_sorted = sorted(entities, key=lambda e: int(e.get("start", 0)))
            marked = ""
            last = 0
            for sp in spans_sorted:
                s = int(sp.get("start", 0))
                e = int(sp.get("end", 0))
                label = sp.get("label", "entity")
                marked += text[last:s]
                entity_text = text[s:e].replace("\n", " ")
                marked += f"[{entity_text}]({label})"
                last = e
            marked += text[last:]
            example = marked
        intents_map.setdefault(intent, []).append(example)

    # Compose YAML
    nlu_section = {"version": "2.0", "nlu": []}
    for intent, examples in intents_map.items():
        block = {"intent": intent, "examples": "|\n" + "\n".join(f"  - {ex}" for ex in examples)}
        nlu_section["nlu"].append(block)

    target = os.path.join(data_dir, "nlu.yml")
    with open(target, "w", encoding="utf-8") as fh:
        yaml.dump(nlu_section, fh, sort_keys=False, allow_unicode=True)

    return target


def _which_rasa_executable():
    """
    Return a command list to invoke rasa in the current environment.
    Prefer running rasa as a module with the same Python interpreter (sys.executable -m rasa)
    so the Flask process uses the same venv where rasa is installed.
    """
    import shutil, sys
    # if there is a system 'rasa' on PATH and it matches our sys.executable, use it;
    # otherwise prefer `sys.executable -m rasa` so it runs inside the same venv.
    rasa_path = shutil.which("rasa")
    if rasa_path:
        try:
            # quick check: run `rasa --version` via PATH rasa to see if it works
            return ["rasa"]
        except Exception:
            pass
    # fallback to module invocation via same Python interpreter
    return [sys.executable, "-m", "rasa"]


def find_latest_rasa_model(rasa_project_path: str):
    models_dir = os.path.join(rasa_project_path, "models")
    if not os.path.isdir(models_dir):
        return None
    gz = sorted(glob(os.path.join(models_dir, "*.tar.gz")), key=os.path.getmtime, reverse=True)
    return gz[0] if gz else None


def train_rasa_model(base_dir: str) -> str:
    """
    Robust Rasa training:
      - writes annotations -> rasa_project/data/nlu.yml (uses annotations_to_rasa_nlu)
      - backs up existing nlu.yml
      - runs `rasa train nlu` using the same Python interpreter (sys.executable -m rasa)
      - saves full logs to backend/models/rasa_model/training_log_{ts}.txt
      - copies produced .tar.gz into backend/models/rasa_model/
      - writes metadata.json and returns dest path
    """
    # allow override with env var for safety
    rasa_project_path = os.environ.get("RASA_PROJECT_PATH")
    if not rasa_project_path:
        # parent directory of backend (your layout: rasa_chatbot/)
        rasa_project_path = os.path.abspath(os.path.join(base_dir, "..", ".."))
    rasa_project_path = os.path.abspath(rasa_project_path)

    annotations_file = os.path.join(base_dir, "data", "annotations.json")
    dest_models_dir = os.path.join(base_dir, "models", "rasa_model")
    os.makedirs(dest_models_dir, exist_ok=True)

    if not os.path.exists(annotations_file):
        raise FileNotFoundError("annotations.json not found at: " + annotations_file)

    # load annotations
    with open(annotations_file, "r", encoding="utf-8") as fh:
        annotations = json.load(fh)

    # convert -> rasa/data/nlu.yml using your converter function
    # annotations_to_rasa_nlu is defined above in same file (keep it)
    nlu_path = annotations_to_rasa_nlu(annotations, rasa_project_path)

    # backup existing nlu.yml (if any)
    nlu_file = os.path.join(rasa_project_path, "data", "nlu.yml")
    try:
        if os.path.exists(nlu_file):
            backup = nlu_file + f".bak_{int(time.time())}"
            shutil.copy2(nlu_file, backup)
    except Exception:
        # not fatal, continue
        pass

    # build command to run rasa; prefer module invocation to use same venv
    rasa_cmd = _which_rasa_executable()
    cmd = rasa_cmd + ["train", "nlu"]  # faster: only NLU
    env = os.environ.copy()

    # run training synchronously and capture logs (so Flask returns clear errors)
    proc = subprocess.run(cmd, cwd=rasa_project_path, env=env, capture_output=True, text=True)
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""

    ts = int(time.time())
    # save training logs for debugging
    log_file = os.path.join(dest_models_dir, f"training_log_{ts}.txt")
    with open(log_file, "w", encoding="utf-8") as lf:
        lf.write("CMD: " + " ".join(cmd) + "\n\n")
        lf.write("CWD: " + rasa_project_path + "\n\n")
        lf.write("=== STDOUT ===\n")
        lf.write(stdout + "\n\n")
        lf.write("=== STDERR ===\n")
        lf.write(stderr + "\n")

    if proc.returncode != 0:
        # raise with pointer to saved log so UI can show where to inspect
        raise RuntimeError(
            "Rasa training failed. See training log: "
            + log_file
            + "\n\nSTDERR:\n"
            + stderr[:4000]
            + "\n\nSTDOUT:\n"
            + stdout[:4000]
        )

    # find produced model in rasa project
    latest = find_latest_rasa_model(rasa_project_path)
    if not latest:
        raise RuntimeError("Rasa trained but no model file found in rasa_project/models. See log: " + log_file)

    # copy to backend models dir
    dest_name = f"model_v{ts}.tar.gz"
    dest_path = os.path.join(dest_models_dir, dest_name)
    shutil.copy2(latest, dest_path)

    # write metadata
    metadata = {
        "info": {"name": "rasa_model", "trained_at": ts, "version": f"v{ts}"},
        "file": dest_name,
        "original_model_path": latest,
        "training_log": log_file,
        "rasa_stdout_snippet": stdout[:4000],
        "rasa_stderr_snippet": stderr[:4000],
    }
    meta_file = os.path.join(dest_models_dir, "metadata.json")
    with open(meta_file, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    return dest_path

import os
import json
import time
from flask import Flask, request, jsonify
from flask_cors import CORS

from utils.tokenizer import tokenize_text
from utils.model_utils import train_spacy_model, train_rasa_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

ANNOTATIONS_FILE = os.path.join(DATA_DIR, 'annotations.json')
INTENTS_FILE = os.path.join(DATA_DIR, 'intents.json')
ENTITIES_FILE = os.path.join(DATA_DIR, 'entities.json')

# ensure data files exist
for f, default in [(ANNOTATIONS_FILE, []), (INTENTS_FILE, []), (ENTITIES_FILE, [])]:
    if not os.path.exists(f):
        with open(f, 'w', encoding='utf-8') as fh:
            json.dump(default, fh, indent=2)

app = Flask(__name__)
CORS(app)

# register new API blueprints (workspace, auth, training, models)
try:
    from api_blueprints.auth_api import bp as auth_bp
    from api_blueprints.workspace_api import bp as ws_bp
    from api_blueprints.train_api import bp as train_bp
    from api_blueprints.models_api import bp as models_bp

    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(ws_bp, url_prefix='/api')
    app.register_blueprint(train_bp, url_prefix='/api')
    app.register_blueprint(models_bp, url_prefix='/api')
except Exception as e:
    # non-fatal: if import fails keep legacy routes working, but print error for debugging
    import traceback, sys
    print('Failed to register API blueprints:', file=sys.stderr)
    traceback.print_exc()


def _load_annotations():
    with open(ANNOTATIONS_FILE, 'r', encoding='utf-8') as fh:
        return json.load(fh)


def _save_annotations(data):
    with open(ANNOTATIONS_FILE, 'w', encoding='utf-8') as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


@app.route('/save_annotation', methods=['POST'])
def save_annotation():
    payload = request.get_json(force=True)
    if not payload or 'text' not in payload:
        return jsonify({'error': 'Invalid payload, missing text'}), 400

    annotations = _load_annotations()
    annotations.append(payload)
    _save_annotations(annotations)
    return jsonify({'status': 'ok', 'saved': payload}), 201


@app.route('/train_model', methods=['POST'])
def train_model():
    payload = request.get_json(force=True) or {}
    backend = payload.get('backend', 'spacy')

    if backend == 'spacy':
        try:
            model_path = train_spacy_model(BASE_DIR)
            return jsonify({'status': 'ok', 'model': model_path}), 200
        except Exception as e:
            return jsonify({'error': 'training_failed', 'details': str(e)}), 500
    elif backend == 'rasa':
        try:
            model_path = train_rasa_model(BASE_DIR)
            return jsonify({'status': 'ok', 'model': model_path}), 200
        except Exception as e:
            return jsonify({'error': 'training_failed', 'details': str(e)}), 500
    else:
        return jsonify({'error': 'unknown_backend'}), 400


@app.route('/model_metadata', methods=['GET'])
def model_metadata():
    metadata = []
    # list models directory
    for root, dirs, files in os.walk(MODELS_DIR):
        # only top-level children
        break
    for name in os.listdir(MODELS_DIR):
        path = os.path.join(MODELS_DIR, name)
        if os.path.isdir(path):
            # look for meta.json or version folders
            meta = {'name': name, 'versions': []}
            for child in os.listdir(path):
                child_path = os.path.join(path, child)
                if os.path.isdir(child_path):
                    meta['versions'].append(child)
                elif child.lower().endswith('.json'):
                    try:
                        with open(child_path, 'r', encoding='utf-8') as fh:
                            meta['info'] = json.load(fh)
                    except Exception:
                        meta['info'] = {'file': child}
            metadata.append(meta)
    return jsonify({'models': metadata})


@app.route('/tokenize', methods=['POST'])
def tokenize():
    payload = request.get_json(force=True) or {}
    text = payload.get('text', '')
    if not text:
        return jsonify({'tokens': []})
    tokens = tokenize_text(text)
    return jsonify({'tokens': tokens})


if __name__ == '__main__':
    # Run on port 5000 as specified
    app.run(host='0.0.0.0', port=5000, debug=True)

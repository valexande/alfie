"""Execute TRUSTED predictors only inside the documented offline container sandbox.

No model substitution, encoder fitting, data synthesis, or training is performed.
Inspection loads executable artifacts too. The container runner is the isolation boundary;
these checks catch common accidental host/unsafe invocations, not hostile containers.
"""
import argparse
import base64
import hashlib
from html.parser import HTMLParser
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require_sandbox():
    if not (Path('/run/.containerenv').exists() or Path('/.dockerenv').exists()):
        raise RuntimeError('Refusing model execution outside a container')
    status = dict(line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
    if os.getuid() == 0 or int(status['CapEff'].strip(), 16) or status['NoNewPrivs'].strip() != '1':
        raise RuntimeError('Require nonroot, dropped capabilities and no-new-privileges')
    if set(os.listdir('/sys/class/net')) - {'lo'}:
        raise RuntimeError('Require --network=none (only loopback may exist)')
    root = next(line.split() for line in Path('/proc/mounts').read_text().splitlines() if line.split()[1] == '/')
    if 'ro' not in root[3].split(','):
        raise RuntimeError('Require --read-only root filesystem')
    limits = {name: (Path('/sys/fs/cgroup') / name).read_text().strip()
              for name in ('cpu.max', 'memory.max', 'memory.swap.max', 'pids.max')}
    quota, period = limits['cpu.max'].split()
    if (quota == 'max' or int(quota) / int(period) > 4
            or limits['memory.max'] == 'max' or int(limits['memory.max']) > 12 * 1024 ** 3
            or limits['pids.max'] == 'max' or int(limits['pids.max']) > 512
            or limits['memory.swap.max'] != '0'):
        raise RuntimeError(f'Require <=4 CPUs, <=12 GiB RAM, no swap and <=512 PIDs: {limits}')
    return {'uid': os.getuid(), 'cap_eff': status['CapEff'].strip(),
            'no_new_privs': status['NoNewPrivs'].strip(), 'interfaces': os.listdir('/sys/class/net'),
            'root_read_only': True, 'cgroup_limits': limits}


class ReportInspection(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text = []
        self.headings = []
        self.images = []
        self.cards = 0
        self._heading = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in ('h1', 'h2', 'h3', 'h4'):
            self._heading = True
        if tag == 'img':
            self.images.append(attrs)
        if 'text-card' in attrs.get('class', '').split():
            self.cards += 1

    def handle_endtag(self, tag):
        if tag in ('h1', 'h2', 'h3', 'h4'):
            self._heading = False

    def handle_data(self, data):
        self.text.append(data)
        if self._heading and data.strip():
            self.headings.append(data.strip())


def inspect_report(html, expect_text=False, problem_type=None, require_importance=False,
                   expected_vocabulary_size=None):
    parsed = ReportInspection()
    parsed.feed(html)
    text = ' '.join(parsed.text)
    required = ['Model Explainability Report', 'Executive Summary']
    if problem_type == 'classification':
        required += ['Accuracy', 'Confusion Matrix']
    elif problem_type == 'regression':
        required += ['RMSE', 'R²']
    if expect_text:
        required += ['Macro F1', 'Balanced Accuracy', 'Confusion Matrix', 'Token-Removal Sensitivity']
        if expected_vocabulary_size is not None:
            required.append(f'{expected_vocabulary_size:,}')
    missing = [value for value in required if value.lower() not in text.lower()]
    if missing or not parsed.images:
        raise RuntimeError(f'Incomplete HTML report: missing={missing}, images={len(parsed.images)}')
    if require_importance or expect_text:
        if ('raw-column feature importance' not in text.lower()
                or 'raw-column feature importance failed' in text.lower()
                or not any(image.get('alt') == 'Raw-column permutation importance' for image in parsed.images)):
            raise RuntimeError('Missing or failed raw-column feature importance in report')
    # A syntactically valid data URI need not contain a valid PNG.
    from io import BytesIO
    from PIL import Image
    for image in parsed.images:
        source = image.get('src', '')
        try:
            if not source.startswith('data:image/png;base64,'):
                raise ValueError('Expected an embedded PNG plot')
            decoded = base64.b64decode(source.split(',', 1)[1], validate=True)
            with Image.open(BytesIO(decoded)) as plot:
                if plot.format != 'PNG':
                    raise ValueError('Plot is not a PNG')
                plot.verify()
        except Exception as exc:
            raise RuntimeError(f'Invalid PNG plot: {image.get("alt", "unnamed")}') from exc
    # Missing explanations must be disclosed, but this real text validation requires successful cards.
    if expect_text and (parsed.cards != 6 or 'Word-removal explanation failed' in text):
        raise RuntimeError(f'Text explanations failed/incomplete: {parsed.cards} cards')
    return parsed, {'headings': parsed.headings, 'embedded_images': len(parsed.images),
                    'text_example_cards': parsed.cards,
                    'method_notes': [line.strip() for line in parsed.text
                                     if any(word in line.lower() for word in ('skipped', 'unavailable', 'failed', 'permutation', 'removal sensitivity'))]}


def json_default(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    if hasattr(value, 'item'):
        return value.item()
    return str(value)


def run(args, result):
    result['isolation'] = require_sandbox()
    result['python'] = sys.version
    result['packages'] = {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}
    check = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True)
    result['pip_check'] = {'returncode': check.returncode, 'stdout': check.stdout, 'stderr': check.stderr}
    check.check_returncode()
    (args.output / 'pip-freeze.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    from fastapi.testclient import TestClient
    from api.app import app
    from xai_core.model_loader import load_model
    from xai_core.explainer_factory import ExplainerFactory
    from xai_core.utils import read_evaluation_csv, resolve_target_column

    with TestClient(app) as client:
        response = client.get('/health')
        response.raise_for_status()
        result['health'] = response.json()
        result['model_sha256'] = sha256(args.model)
        start = time.monotonic()
        info = load_model(args.model)
        result['load_seconds'] = time.monotonic() - start
        model = info.model
        if not info.is_autogluon or info.model_type != 'tabular':
            raise RuntimeError('This validator requires the supplied AutoGluon tabular predictor')
        result['model'] = {'label': model.label, 'original_features': model.features(feature_stage='original'),
                           'problem_type': model.problem_type, 'class_labels': model.class_labels,
                           'model_best': model.model_best, 'model_names': model.model_names(),
                           'selected_path': model.path, 'eval_metric': str(model.eval_metric),
                           'mapping_present': info.text_feature_mapping is not None}
        result['model']['best_model_info'] = model.info()['model_info'][model.model_best]
        result['load_passed'] = True
        # Persist load evidence even if evaluation fails or times out later.
        save(args, result)
        if args.inspect_only:
            result['status'] = 'load-only; inference/report not tested'
            return
        result['dataset'] = {'path': str(args.data), 'sha256': sha256(args.data), 'provenance': args.provenance,
                             'held_out_status': args.held_out_status}
        df = read_evaluation_csv(args.data.read_bytes())
        if args.expected_rows is not None and len(df) != args.expected_rows:
            raise RuntimeError(f'Expected {args.expected_rows} evaluation rows; got {len(df)}')
        target = resolve_target_column(df, args.target, model_label=model.label)
        explainer = ExplainerFactory.create(model, df.drop(columns=[target]), df[target],
                                            model_type=info.model_type, label=target,
                                            text_feature_mapping=info.text_feature_mapping, max_samples=args.max_samples)
        start = time.monotonic()
        predictions = explainer.get_predictions()
        probabilities = explainer.get_prediction_probabilities()
        result['predict_and_proba_seconds'] = time.monotonic() - start
        result['metrics'] = explainer.get_metrics()
        start = time.monotonic()
        importance = explainer.get_feature_importance()
        if importance is None or importance.empty:
            raise RuntimeError('Raw-column feature importance is required for these runtime validation cases')
        result['raw_column_importance'] = importance.to_dict(orient='records')
        result['importance_seconds'] = time.monotonic() - start
        result['importance_notes'] = list(explainer.explanation_notes)
        result['prediction_rows'] = len(predictions)
        result['probability_shape'] = list(probabilities.shape) if probabilities is not None else None
        import pandas as pd
        rows = pd.DataFrame({'actual': df[target].to_numpy(), 'predicted': predictions})
        if probabilities is not None:
            for i, label in enumerate(explainer.classes):
                rows[f'probability_{label}'] = probabilities[:, i]
        rows.to_csv(args.output / 'predictions.csv', index=False)
        result['inference_passed'] = True
        save(args, result)
        start = time.monotonic()
        # Real multipart route: no monkeypatches or replacement models/explainers.
        with args.model.open('rb') as model_stream, args.data.open('rb') as data_stream:
            response = client.post('/explain-model', files={
                'model_file': (args.model.name, model_stream, 'application/zip'),
                'data_file': (args.data.name, data_stream, 'text/csv')},
                data={'target_col': target, 'user_level': 'expert', 'max_shap_samples': args.max_samples})
        result['report_seconds'] = time.monotonic() - start
        result['report_http_status'] = response.status_code
        if response.status_code != 200:
            (args.output / 'report-error.txt').write_text(response.text)
            response.raise_for_status()
        if 'text/html' not in response.headers.get('content-type', ''):
            raise RuntimeError('Report endpoint did not return HTML')
        report = args.output / 'report.html'
        report.write_text(response.text)
        result['report_sha256'] = sha256(report)
        vocabulary_size = args.expected_vocabulary_size
        if vocabulary_size is None and info.text_feature_mapping is not None:
            vocabulary_size = sum(len(group['feature_names'])
                                  for group in info.text_feature_mapping['ngram_features'].values())
        parsed, result['html_inspection'] = inspect_report(
            response.text, args.expect_text_explanations, info.problem_type,
            require_importance=True, expected_vocabulary_size=vocabulary_size
        )
        # Extract the actual generated plot images for independent visual review.
        for i, image in enumerate(parsed.images):
            source = image.get('src', '')
            if source.startswith('data:image/png;base64,'):
                (args.output / f'report-plot-{i:02}.png').write_bytes(base64.b64decode(source.split(',', 1)[1], validate=True))
        result['report_passed'] = True
        result['status'] = 'passed'


def save(args, result):
    (args.output / 'validation.json').write_text(json.dumps(result, indent=2, default=json_default, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True, help='Trusted model ZIP; root selection stays in the application loader')
    parser.add_argument('--data', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--target')
    parser.add_argument('--provenance')
    parser.add_argument('--held-out-status', choices=['supplied-test-split; not independently audited', 'unknown; compatibility only'])
    parser.add_argument('--expected-rows', type=int)
    parser.add_argument('--max-samples', type=int, default=300)
    parser.add_argument('--inspect-only', action='store_true')
    parser.add_argument('--expect-text-explanations', action='store_true',
                        help='Require six successful word-removal explanation examples')
    parser.add_argument('--expected-vocabulary-size', type=int,
                        help='Expected vocabulary size; defaults to the validated model mapping')
    args = parser.parse_args()
    if not args.inspect_only and not all((args.data, args.provenance, args.held_out_status)):
        parser.error('Evaluation requires --data, --provenance and --held-out-status')
    if not 10 <= args.max_samples <= 5000:
        parser.error('--max-samples must be between 10 and 5000')
    if args.expected_vocabulary_size is not None and args.expected_vocabulary_size < 0:
        parser.error('--expected-vocabulary-size cannot be negative')
    args.output.mkdir(parents=True, exist_ok=True)
    result = {'status': 'running', 'load_passed': False, 'inference_passed': False, 'report_passed': False}
    start = time.monotonic()
    try:
        run(args, result)
    except Exception:
        result['status'] = 'failed'
        result['traceback'] = traceback.format_exc()
        print(result['traceback'], file=sys.stderr)
        return 1
    finally:
        result['total_seconds'] = time.monotonic() - start
        save(args, result)
    return 0


if __name__ == '__main__':
    sys.exit(main())

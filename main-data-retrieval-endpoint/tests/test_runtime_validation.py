"""Synthetic-only checks of runtime evidence validation and passive ZIP staging."""
import base64
import io
import os
from pathlib import Path
import stat
import subprocess
import sys
import zipfile

from PIL import Image

import pytest

from scripts.stage_xai_artifacts import audit_zip, stage
from scripts.validate_runtime import inspect_report, require_sandbox


def archive(name, content=b'x', mode=None):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as z:
        member = zipfile.ZipInfo(name)
        if mode is not None:
            member.external_attr = mode << 16
        z.writestr(member, content)
    stream.seek(0)
    return zipfile.ZipFile(stream)


@pytest.mark.parametrize('name', ['../predictor.pkl', '/predictor.pkl', 'a\\predictor.pkl',
                                   'a/../predictor.pkl', 'a//predictor.pkl', './predictor.pkl', 'C:/predictor.pkl'])
def test_staging_rejects_unsafe_zip_paths(name):
    with archive(name) as z, pytest.raises(ValueError, match='Unsafe'):
        audit_zip(z)


def test_staging_rejects_symlinks():
    with archive('predictor.pkl', mode=stat.S_IFLNK | 0o777) as z, pytest.raises(ValueError, match='Nonregular'):
        audit_zip(z)


def test_staging_accepts_regular_member_without_loading_it():
    with archive('predictor.pkl', b'not a pickle') as z:
        assert audit_zip(z)['members'] == 1


def test_staging_rejects_duplicate_destinations():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as z:
        z.writestr('a/', b'')
        z.writestr('a', b'x')
    stream.seek(0)
    with zipfile.ZipFile(stream) as z, pytest.raises(ValueError, match='Duplicate'):
        audit_zip(z)


def report(cards=6):
    image = io.BytesIO()
    Image.new('RGB', (2, 2), 'white').save(image, format='PNG')
    png = base64.b64encode(image.getvalue()).decode()
    return ('<h1>Model Explainability Report</h1><h2>Executive Summary</h2>'
            '<h2>Macro F1</h2><h2>Balanced Accuracy</h2><h2>Confusion Matrix</h2>'
            '<h2>Token-Removal Sensitivity</h2><p>2,363 vocabulary terms</p>'
            f'<img src="data:image/png;base64,{png}" alt="Confusion Matrix">'
            '<h2>Raw-Column Feature Importance</h2>'
            f'<img src="data:image/png;base64,{png}" alt="Raw-column permutation importance">' +
            '<div class="text-card">example</div>' * cards)


def test_validate_runtime_inspects_real_section_contract():
    parsed, evidence = inspect_report(report(), expect_text=True, expected_vocabulary_size=2363)
    assert evidence['text_example_cards'] == 6
    assert evidence['embedded_images'] == 2
    assert parsed.images[0]['alt'] == 'Confusion Matrix'


@pytest.mark.parametrize('html', ['', '<html>Success</html>', report(0), report(5),
                                 report().replace('2,363', 'unknown'),
                                 report() + 'Word-removal explanation failed for row 1'])
def test_validate_runtime_rejects_empty_or_incomplete_text_reports(html):
    with pytest.raises(RuntimeError):
        inspect_report(html, expect_text=True, expected_vocabulary_size=2363)


def test_text_report_vocabulary_check_is_configurable():
    html = report().replace('2,363 vocabulary terms', '17 vocabulary terms')
    inspect_report(html, expect_text=True, expected_vocabulary_size=17)
    with pytest.raises(RuntimeError, match='Incomplete'):
        inspect_report(html, expect_text=True, expected_vocabulary_size=18)


def test_staging_uses_explicit_bundle_members_and_neutral_output_names(tmp_path):
    def zip_bytes(members):
        output = io.BytesIO()
        with zipfile.ZipFile(output, 'w') as archive:
            for name, content in members.items():
                archive.writestr(name, content)
        return output.getvalue()

    predictor = zip_bytes({'predictor.pkl': b'not executable data', 'metadata.json': b'{}'})
    csv = b'labels,text\n1,hello\n'
    dataset = zip_bytes({'evaluation/input.csv': csv})
    bundle = tmp_path / 'bundle.zip'
    bundle.write_bytes(zip_bytes({'models/model.zip': predictor, 'data/splits.zip': dataset}))
    output = tmp_path / 'staged'
    result = stage(bundle, output, model_member='models/model.zip',
                   data_member='data/splits.zip', csv_member='evaluation/input.csv')
    assert (output / 'predictor.zip').read_bytes() == predictor
    assert (output / 'evaluation.csv').read_bytes() == csv
    assert result['dataset_provenance'] == 'bundle.zip!data/splits.zip!evaluation/input.csv'


@pytest.mark.parametrize('html', [
    report().replace('Raw-column permutation importance', 'absent'),
    report().replace('Raw-Column Feature Importance', 'absent'),
    report() + '<p>Native raw-column feature importance failed: backend error</p>',
])
def test_validator_rejects_missing_or_failed_importance(html):
    with pytest.raises(RuntimeError, match='importance'):
        inspect_report(html, expect_text=True)


def test_validator_rejects_corrupt_png_data():
    import re
    html = re.sub(r'data:image/png;base64,[^\"]+', 'data:image/png;base64,eA==', report(), count=1)
    with pytest.raises(RuntimeError, match='Invalid PNG'):
        inspect_report(html, expect_text=True)


@pytest.mark.parametrize('mode', ['create_failure', 'bad_data', 'success'])
def test_runner_cleans_only_an_id_it_successfully_created(tmp_path, mode):
    # Fake Podman: exercise shell control flow without containers or model execution.
    log = tmp_path / 'calls.log'
    podman = tmp_path / 'podman'
    podman.write_text(f'''#!{sys.executable}
import os, sys
from pathlib import Path
with Path(os.environ['FAKE_LOG']).open('a') as stream:
    stream.write(' '.join(sys.argv[1:]) + '\\n')
command = sys.argv[1]
if command == 'info': print('true')
elif command == 'create':
    if os.environ['FAKE_MODE'] == 'create_failure': sys.exit(17)
    print('a' * 64)
elif command == 'inspect': print('0' if '--format' in sys.argv else '{{}}')
''')
    podman.chmod(0o755)
    model, data = tmp_path / 'model.zip', tmp_path / 'data.csv'
    model.write_bytes(b'not a model')
    if mode != 'bad_data':
        data.write_bytes(b'label,text\n1,hi')
    env = dict(os.environ, PATH=str(tmp_path) + os.pathsep + os.environ['PATH'],
               FAKE_LOG=str(log), FAKE_MODE=mode)
    runner = Path(__file__).resolve().parents[1] / 'scripts/run_isolated_validation.sh'
    result = subprocess.run(['bash', str(runner), 'fake-image', str(model), str(data), str(tmp_path / 'output')],
                            env=env, capture_output=True, text=True, timeout=15)
    removals = [line for line in log.read_text().splitlines() if line.startswith('rm ')]
    if mode == 'success':
        assert result.returncode == 0, result.stderr
        assert removals == ['rm -f ' + 'a' * 64]
    else:
        assert result.returncode != 0
        assert removals == []


def test_validate_runtime_refuses_host_execution(monkeypatch):
    from pathlib import Path
    real_exists = Path.exists
    monkeypatch.setattr(Path, 'exists', lambda p: False if str(p) in ('/run/.containerenv', '/.dockerenv') else real_exists(p))
    with pytest.raises(RuntimeError, match='outside a container'):
        require_sandbox()

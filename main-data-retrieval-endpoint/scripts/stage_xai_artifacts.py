"""Audit an XAI evaluation bundle and stage its predictor ZIP and evaluation CSV.

Uses stdlib ZIP/JSON only: never deserializes model objects. Existing outputs are not
replaced. This bounded archive audit is not an untrusted-upload API sandbox.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile

MAX_BYTES = 1024 * 1024 * 1024
MAX_MEMBERS = 20000


def audit_zip(archive):
    members = archive.infolist()
    if len(members) > MAX_MEMBERS or sum(i.file_size for i in members) > MAX_BYTES:
        raise ValueError('ZIP member count or uncompressed size exceeds staging budget')
    seen = set()
    for member in members:
        name = member.filename
        path = PurePosixPath(name)
        if (not name or name.startswith('/') or '\\' in name or ':' in name
                or any(p in ('', '.', '..') for p in name.rstrip('/').split('/'))):
            raise ValueError(f'Unsafe ZIP member path: {name!r}')
        if str(path) in seen:
            raise ValueError(f'Duplicate ZIP destination: {name!r}')
        seen.add(str(path))
        kind = stat.S_IFMT(member.external_attr >> 16)
        if kind not in (0, stat.S_IFREG, stat.S_IFDIR) or member.flag_bits & 1:
            raise ValueError(f'Nonregular/encrypted ZIP member: {name!r}')
    bad = archive.testzip()
    if bad is not None:
        raise ValueError(f'ZIP CRC failed: {bad}')
    return {'members': len(members), 'uncompressed_bytes': sum(i.file_size for i in members),
            'max_compression_ratio': max((i.file_size / max(i.compress_size, 1) for i in members), default=0)}


def stage(outer_path, output, *, model_member, data_member, csv_member):
    output.mkdir(parents=True, exist_ok=False)
    result = {'outer_path': str(outer_path), 'outer_sha256': hashlib.sha256(outer_path.read_bytes()).hexdigest()}
    with zipfile.ZipFile(outer_path) as outer:
        result['outer_audit'] = audit_zip(outer)
        model_bytes = outer.read(model_member)
        data_bytes = outer.read(data_member)
    with zipfile.ZipFile(io.BytesIO(model_bytes)) as model:
        result['predictor_audit'] = audit_zip(model)
        roots = [str(PurePosixPath(i.filename).parent) for i in model.infolist() if PurePosixPath(i.filename).name == 'predictor.pkl']
        result['predictor_roots'] = roots
        if '.' not in roots:
            raise ValueError('Expected predictor.pkl at archive root; no clone substitution allowed')
        result['model_metadata'] = json.loads(model.read('metadata.json'))
    with zipfile.ZipFile(io.BytesIO(data_bytes)) as dataset:
        result['dataset_audit'] = audit_zip(dataset)
        csv_bytes = dataset.read(csv_member)
    # Preserve the original inner ZIP byte-for-byte, including its nested clone.
    for name, content in [('predictor.zip', model_bytes), ('evaluation.csv', csv_bytes)]:
        (output / name).write_bytes(content)
        result[name] = {'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)}
    result['dataset_provenance'] = f'{outer_path.name}!{data_member}!{csv_member}'
    (output / 'staging-provenance.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('outer', type=Path)
    parser.add_argument('output', type=Path, help='New dedicated directory')
    parser.add_argument('--model-member', required=True, help='Predictor ZIP member inside the bundle')
    parser.add_argument('--data-member', required=True, help='Dataset ZIP member inside the bundle')
    parser.add_argument('--csv-member', required=True, help='Evaluation CSV member inside the dataset ZIP')
    args = parser.parse_args()
    print(json.dumps(stage(args.outer, args.output, model_member=args.model_member,
                           data_member=args.data_member, csv_member=args.csv_member), indent=2))

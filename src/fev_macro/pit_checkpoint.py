"""Pinned foundation-model evidence and explicit research-use restrictions.

A publisher's dated immutable checkpoint establishes that the weights existed at
an origin. This is NOT a claim to have audited every pretraining observation.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re

import pandas as pd

from .pit import PITError, content_hash, information_date


# The adapter, not an editable manifest, determines the permitted publisher and files.
CHECKPOINT_SPECS = {
    'chronos2': dict(repository='amazon/chronos-2',
                     files=('config.json', 'model.safetensors'), license='Apache-2.0', research_only=False),
    'tabpfn_bridge': dict(repository='Prior-Labs/tabpfn_3',
        files=('config.json', 'tabpfn-v3-regressor-v3_default.ckpt', 'LICENSE'),
        license='TabPFN-3 non-commercial', research_only=True),
    'tabpfn_ts': dict(repository='Prior-Labs/tabpfn_3',
        files=('config.json', 'tabpfn-v3-regressor-v3_20260506_timeseries.ckpt', 'LICENSE'),
        license='TabPFN-3 non-commercial', research_only=True),
    'timesfm3': dict(repository='google/timesfm-3.0-pytorch',
        files=('config.json', 'model.safetensors', 'LICENSE'),
        license='TimesFM Non-Commercial License v1.0; non-production', research_only=True),
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoint(manifest_path, origin, *, model='chronos2', model_use='production'):
    if model_use not in {'research', 'production'}:
        raise PITError('model_use must be research or production')
    if model not in CHECKPOINT_SPECS:
        raise PITError(f'Unknown checkpoint specification: {model}')
    spec = CHECKPOINT_SPECS[model]
    if spec['research_only'] and model_use != 'research':
        raise PITError(f"{model} requires model_use=research: {spec['license']}")
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict):
        raise PITError('Checkpoint manifest must be an object')
    if (manifest.get('repository') != spec['repository'] or not isinstance(manifest.get('revision'), str)
            or not re.fullmatch('[0-9a-f]{40}', manifest['revision'])):
        raise PITError(f"{model} requires an immutable {spec['repository']} commit")
    evidence = manifest.get('publication_evidence')
    if not isinstance(evidence, dict) or evidence.get('commit_id') != manifest['revision']:
        raise PITError('Checkpoint publication evidence does not match revision')
    published = pd.Timestamp(evidence.get('created_at'))
    if pd.isna(published) or published.tzinfo is None:
        raise PITError('Checkpoint publication timestamp must include timezone')
    if published.tz_convert('America/New_York').date() > information_date(origin).date():
        raise PITError('Checkpoint was not available before the forecast origin')
    expected_url = f"https://huggingface.co/{spec['repository']}/commit/{manifest['revision']}"
    if evidence.get('source_url') != expected_url:
        raise PITError('Missing publisher commit evidence URL')
    if not isinstance(manifest.get('directory'), str):
        raise PITError('Missing checkpoint directory')
    directory = (manifest_path.parent / manifest['directory']).resolve()
    files = manifest.get('files')
    if not isinstance(files, dict) or set(files) != set(spec['files']):
        raise PITError(f"Exact checkpoint files required: {spec['files']}")
    # Only these files may be loaded; reject unrecorded alternate weights/configs.
    if {p.name for p in directory.iterdir()} != set(files):
        raise PITError('Checkpoint directory contains unrecorded files')
    for name, record in files.items():
        if not isinstance(record, dict) or not isinstance(record.get('sha256'), str):
            raise PITError(f'Missing checkpoint file hash: {name}')
        path = directory / name
        if file_sha256(path) != record['sha256']:
            raise PITError(f'Checkpoint hash mismatch: {name}')
        if name.endswith(('.safetensors', '.ckpt')) and record['sha256'] != record.get('publisher_lfs_sha256'):
            raise PITError('Checkpoint does not match publisher LFS hash')
        if not name.endswith(('.safetensors', '.ckpt')):
            body = path.read_bytes()
            git_blob = hashlib.sha1(b'blob ' + str(len(body)).encode() + b'\0' + body).hexdigest()
            if git_blob != record.get('publisher_git_blob'):
                raise PITError(f'Checkpoint file does not match publisher Git blob: {name}')
    return directory, dict(manifest=manifest, manifest_sha256=content_hash(manifest),
                           license=spec['license'], research_only=spec['research_only'], model_use=model_use,
                           availability_basis='publisher immutable commit timestamp and artifact hashes; pretraining corpus not independently audited')

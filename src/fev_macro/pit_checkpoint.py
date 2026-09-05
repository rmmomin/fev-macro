"""Pinned Chronos evidence: publication before origin and exact local artifact hashes.

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


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoint(manifest_path, origin):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('repository') != 'amazon/chronos-2' or not re.fullmatch('[0-9a-f]{40}', manifest.get('revision', '')):
        raise PITError('Chronos requires an immutable amazon/chronos-2 commit')
    evidence = manifest['publication_evidence']
    if evidence['commit_id'] != manifest['revision']:
        raise PITError('Checkpoint publication evidence does not match revision')
    published = pd.Timestamp(evidence['created_at'])
    if published.tzinfo is None:
        raise PITError('Checkpoint publication timestamp must include timezone')
    if published.tz_convert('America/New_York').date() > information_date(origin).date():
        raise PITError('Checkpoint was not available before the forecast origin')
    expected_url = f"https://huggingface.co/amazon/chronos-2/commit/{manifest['revision']}"
    if evidence.get('source_url') != expected_url:
        raise PITError('Missing publisher commit evidence URL')
    directory = (manifest_path.parent / manifest['directory']).resolve()
    files = manifest['files']
    if set(files) != {'config.json', 'model.safetensors'}:
        raise PITError('Exact checkpoint config and safetensors hashes required')
    # Only these files may be loaded; reject unrecorded alternate weights/configs.
    if {p.name for p in directory.iterdir() if p.is_file()} != set(files):
        raise PITError('Checkpoint directory contains unrecorded files')
    for name, record in files.items():
        path = directory / name
        if file_sha256(path) != record['sha256']:
            raise PITError(f'Checkpoint hash mismatch: {name}')
        if name.endswith('.safetensors') and record['sha256'] != record['publisher_lfs_sha256']:
            raise PITError('Checkpoint does not match publisher LFS hash')
        if name == 'config.json':
            body = path.read_bytes()
            git_blob = hashlib.sha1(b'blob ' + str(len(body)).encode() + b'\0' + body).hexdigest()
            if git_blob != record['publisher_git_blob']:
                raise PITError('Checkpoint config does not match publisher Git blob')
    return directory, dict(manifest=manifest, manifest_sha256=content_hash(manifest),
                           availability_basis='publisher immutable commit timestamp and artifact hashes; pretraining corpus not independently audited')

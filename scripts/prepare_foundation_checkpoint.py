#!/usr/bin/env python3
"""Prepare an immutable foundation checkpoint; inference never downloads weights."""
from pathlib import Path
import argparse
import json
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fev_macro.pit import information_date
from fev_macro.pit_checkpoint import CHECKPOINT_SPECS, file_sha256, validate_checkpoint


def prepare_checkpoint(model, revision, origin, out, *, model_use='production', cache_dir=None):
    from huggingface_hub import HfApi, hf_hub_download
    import pandas as pd
    spec = CHECKPOINT_SPECS[model]
    if not re.fullmatch('[0-9a-f]{40}', revision):
        raise ValueError('An immutable 40-character publisher commit is required')
    if spec['research_only'] and model_use != 'research':
        raise ValueError(f"{model} requires --model-use research: {spec['license']}")
    out = Path(out).resolve()
    if out.exists():
        raise ValueError('Use a new directory; checkpoint evidence is never overwritten')
    repo, api = spec['repository'], HfApi()
    matches = [c for c in api.list_repo_commits(repo, revision=revision) if c.commit_id == revision]
    if len(matches) != 1:
        raise ValueError('Revision has no publisher commit timestamp')
    commit = matches[0]
    if pd.Timestamp(commit.created_at).tz_convert('America/New_York').date() > information_date(origin).date():
        raise ValueError('Checkpoint was not available before the forecast origin')
    info = api.model_info(repo, revision=revision, files_metadata=True)
    published = {s.rfilename: s for s in info.siblings}
    if set(spec['files']) - set(published):
        raise ValueError('Required adapter files are absent at this revision')
    # Download all files to the Hub cache first; a failed download cannot create a valid manifest.
    sources = {name: hf_hub_download(repo, name, revision=revision, cache_dir=cache_dir) for name in spec['files']}
    for name, src in sources.items():
        item = published[name]
        if item.lfs and file_sha256(src) != item.lfs.sha256:
            raise ValueError(f'Publisher hash mismatch in cached download: {name}; retry with a fresh --cache-dir')
    directory = out / 'checkpoint'
    directory.mkdir(parents=True)
    files = {}
    for name, src in sources.items():
        dst, item = directory / name, published[name]
        shutil.copyfile(src, dst)
        files[name] = dict(sha256=file_sha256(dst), publisher_git_blob=item.blob_id,
                          publisher_lfs_sha256=item.lfs.sha256 if item.lfs else None)
    manifest = dict(repository=repo, revision=revision, directory='checkpoint', files=files,
        publication_evidence=dict(commit_id=revision, created_at=commit.created_at.isoformat(),
            title=commit.title, source_url=f'https://huggingface.co/{repo}/commit/{revision}'))
    path = out / 'manifest.json'
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    validate_checkpoint(path, origin, model=model, model_use=model_use)
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, choices=CHECKPOINT_SPECS)
    p.add_argument('--revision', required=True)
    p.add_argument('--origin', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--model-use', choices=['research', 'production'], default='production')
    p.add_argument('--cache-dir', help='Optional separate Hub cache, useful after a corrupted download')
    args = p.parse_args()
    try:
        print(f'Validated checkpoint evidence: {prepare_checkpoint(**vars(args))}')
    except (ValueError, OSError, RuntimeError) as exc:
        p.error(str(exc))


if __name__ == '__main__':
    main()

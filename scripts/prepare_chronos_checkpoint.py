#!/usr/bin/env python3
"""Download an explicitly pinned public Chronos checkpoint and archive its evidence."""
from pathlib import Path
import argparse
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fev_macro.pit_checkpoint import file_sha256, validate_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', required=True, help='Immutable 40-character publisher commit')
    parser.add_argument('--origin', required=True, help='Earliest intended forecast origin')
    parser.add_argument('--out', required=True, help='New checkpoint evidence directory')
    args = parser.parse_args()
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    commits = api.list_repo_commits('amazon/chronos-2')
    matches = [c for c in commits if c.commit_id == args.revision]
    if len(matches) != 1:
        parser.error('Revision has no publisher commit timestamp')
    import pandas as pd
    from fev_macro.pit import information_date
    if pd.Timestamp(matches[0].created_at).tz_convert('America/New_York').date() > information_date(args.origin).date():
        parser.error('Checkpoint was not available before the forecast origin')
    info = api.model_info('amazon/chronos-2', revision=args.revision, files_metadata=True)
    out = Path(args.out).resolve()
    if out.exists():
        parser.error('Use a new directory; existing checkpoint evidence is never overwritten')
    directory = out / 'checkpoint'
    directory.mkdir(parents=True)
    files = {}
    for item in info.siblings:
        if item.rfilename not in {'config.json', 'model.safetensors'}:
            continue
        src = hf_hub_download('amazon/chronos-2', item.rfilename, revision=args.revision)
        dst = directory / item.rfilename
        shutil.copyfile(src, dst)
        files[item.rfilename] = dict(sha256=file_sha256(dst), publisher_git_blob=item.blob_id,
                                      publisher_lfs_sha256=item.lfs.sha256 if item.lfs else None)
    commit = matches[0]
    manifest = dict(repository='amazon/chronos-2', revision=args.revision, directory='checkpoint', files=files,
                    publication_evidence=dict(commit_id=commit.commit_id, created_at=commit.created_at.isoformat(),
                        title=commit.title, source_url=f'https://huggingface.co/amazon/chronos-2/commit/{commit.commit_id}'))
    path = out / 'manifest.json'
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    validate_checkpoint(path, args.origin)
    print(f'Validated checkpoint evidence: {path}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path


FRED_VINTAGE_URLS: tuple[str, ...] = (
    "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/historical-vintages-of-fred-md-2015-01-to-2024-12.zip?sc_lang=en&hash=831F98A7EC8D3809881DF067965B50FF",
    "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/historical-vintages-of-fred-qd-2018-05-to-2024-12.zip?sc_lang=en&hash=4088DF99A1CCB4F6ED49F5B88A7C636D",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and extract historical FRED-MD/FRED-QD vintages.")
    parser.add_argument("--output_dir", type=str, default="data/historical")
    parser.add_argument("--downloads_dir", type=str, default=None, help="Optional zip download directory.")
    parser.add_argument("--force", action="store_true", help="Re-download archives and overwrite extracted folders.")
    return parser.parse_args()


def _filename_from_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    name = Path(parsed.path).name
    if not name:
        raise ValueError(f"Unable to infer filename from URL: {url}")
    return name


def _download(url: str, destination: Path, force: bool = False) -> Path:
    if destination.exists() and not force:
        print(f"Skipping existing download: {destination}")
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading: {url}")
    with urllib.request.urlopen(url) as response:
        destination.write_bytes(response.read())
    print(f"Saved: {destination}")
    return destination


def _extract(zip_path: Path, output_dir: Path, force: bool = False) -> list[Path]:
    extracted_roots: list[Path] = []
    with zipfile.ZipFile(zip_path, "r") as archive:
        members = [m for m in archive.namelist() if m and not m.endswith("/")]
        top_level = sorted({Path(m).parts[0] for m in members if Path(m).parts})

        if force:
            for name in top_level:
                candidate = output_dir / name
                if candidate.exists():
                    if candidate.is_dir():
                        shutil.rmtree(candidate)
                    else:
                        candidate.unlink()

        archive.extractall(output_dir)
        extracted_roots = [output_dir / name for name in top_level]

    print(f"Extracted: {zip_path} -> {output_dir}")
    return extracted_roots


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    downloads_dir = (
        Path(args.downloads_dir).expanduser().resolve()
        if args.downloads_dir is not None
        else (output_dir / "downloads")
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    downloads_dir.mkdir(parents=True, exist_ok=True)

    all_extracted: list[Path] = []
    for url in FRED_VINTAGE_URLS:
        filename = _filename_from_url(url)
        zip_path = downloads_dir / filename
        _download(url=url, destination=zip_path, force=args.force)
        extracted = _extract(zip_path=zip_path, output_dir=output_dir, force=args.force)
        all_extracted.extend(extracted)

    print("Completed historical vintage download/extract.")
    if all_extracted:
        print("Extracted folders:")
        for path in sorted({p.resolve() for p in all_extracted}):
            print(f" - {path}")


if __name__ == "__main__":
    main()

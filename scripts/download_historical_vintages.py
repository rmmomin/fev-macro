#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import re
import shutil
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path


FRED_VINTAGE_URLS: tuple[str, ...] = (
    "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/historical_fred-md.zip?sc_lang=en&hash=8A23C5FAF7A0D743A353D77DF4704028",
    "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/historical-vintages-of-fred-md-2015-01-to-2024-12.zip?sc_lang=en&hash=831F98A7EC8D3809881DF067965B50FF",
    "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/historical-vintages-of-fred-qd-2018-05-to-2024-12.zip?sc_lang=en&hash=4088DF99A1CCB4F6ED49F5B88A7C636D",
)
FRED_DATABASES_PAGE_URL = "https://www.stlouisfed.org/research/economists/mccracken/fred-databases"
DEFAULT_MD_DIR = "data/historical/md/vintages_1999_2026"
DEFAULT_QD_DIR = "data/historical/qd/vintages_2018_2026"
MDFILE_PATTERN = re.compile(r"(?:fred[-_]?md_(\d{4})m(\d{1,2})|(\d{4})-(\d{2}))\.csv$", re.IGNORECASE)
QDFILE_PATTERN = re.compile(r"fred[-_]?qd_(\d{4})m(\d{1,2})\.csv$", re.IGNORECASE)
MONTHLY_URL_PATTERN = re.compile(
    r'href="(?P<href>[^"]*?/fred-md/monthly/[^"]+?\.csv(?:\?[^"]*)?)"',
    re.IGNORECASE,
)
QUARTERLY_URL_PATTERN = re.compile(
    r'href="(?P<href>[^"]*?/fred-md/quarterly/[^"]+?\.csv(?:\?[^"]*)?)"',
    re.IGNORECASE,
)
MONTHLY_URL_PERIOD_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"/monthly/(?P<year>\d{4})-(?P<month>\d{2})(?:-md)?\.csv", re.IGNORECASE),
    re.compile(r"/monthly/fred-md_(?P<year>\d{4})m(?P<month>\d{1,2})\.csv", re.IGNORECASE),
)
QUARTERLY_URL_PERIOD_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"/quarterly/(?P<year>\d{4})-(?P<month>\d{2})(?:-qd)?\.csv", re.IGNORECASE),
    re.compile(r"/quarterly/fred-qd_(?P<year>\d{4})m(?P<month>\d{1,2})\.csv", re.IGNORECASE),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and extract historical FRED-MD/FRED-QD vintages.")
    parser.add_argument("--output_dir", type=str, default="data/historical")
    parser.add_argument("--downloads_dir", type=str, default=None, help="Optional zip download directory.")
    parser.add_argument("--force", action="store_true", help="Re-download archives and overwrite extracted folders.")
    parser.add_argument(
        "--monthly_output_dir",
        type=str,
        default=DEFAULT_MD_DIR,
        help="Directory where scraped monthly FRED-MD vintages are stored as YYYY-MM.csv.",
    )
    parser.add_argument(
        "--quarterly_output_dir",
        type=str,
        default=DEFAULT_QD_DIR,
        help="Directory where scraped quarterly FRED-QD vintages are stored as FRED-QD_YYYYmM.csv.",
    )
    parser.add_argument(
        "--fred_page_url",
        type=str,
        default=FRED_DATABASES_PAGE_URL,
        help="FRED databases page used to scrape quarterly vintage CSV links.",
    )
    parser.add_argument("--skip_archives", action="store_true", help="Skip zip archive download/extract step.")
    parser.add_argument("--skip_monthly", action="store_true", help="Skip monthly CSV scrape/download step.")
    parser.add_argument("--skip_quarterly", action="store_true", help="Skip quarterly CSV scrape/download step.")
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


def _archive_extract_root(output_dir: Path, url: str) -> Path:
    lower = url.lower()
    if "fred-qd" in lower:
        return output_dir / "archives" / "qd"
    return output_dir / "archives" / "md"


def _extract(zip_path: Path, output_dir: Path, force: bool = False) -> list[Path]:
    extracted_roots: list[Path] = []
    with zipfile.ZipFile(zip_path, "r") as archive:
        members = [m for m in archive.namelist() if m and not m.endswith("/")]
        top_level = sorted({Path(m).parts[0] for m in members if Path(m).parts})
        has_root_level_files = any(len(Path(m).parts) == 1 for m in members)

        # Some St. Louis Fed archives unpack flat files at root. Put these in a
        # zip-specific subfolder to avoid polluting data/historical/.
        extract_dir = output_dir / zip_path.stem if has_root_level_files else output_dir

        if force:
            if extract_dir.exists():
                if extract_dir.is_dir():
                    shutil.rmtree(extract_dir)
                else:
                    extract_dir.unlink()
            for name in top_level:
                candidate = extract_dir / name
                if candidate.exists():
                    if candidate.is_dir():
                        shutil.rmtree(candidate)
                    else:
                        candidate.unlink()

        extract_dir.mkdir(parents=True, exist_ok=True)
        archive.extractall(extract_dir)
        if has_root_level_files:
            extracted_roots = [extract_dir]
        else:
            extracted_roots = [extract_dir / name for name in top_level]

    print(f"Extracted: {zip_path} -> {extract_dir}")
    return extracted_roots


def _normalize_qd_filename(year: int, month: int) -> str:
    return f"FRED-QD_{year:04d}m{month}.csv"


def _normalize_md_filename(year: int, month: int) -> str:
    return f"{year:04d}-{month:02d}.csv"


def _extract_vintage_year_month(url: str, patterns: tuple[re.Pattern[str], ...]) -> tuple[int, int] | None:
    for pattern in patterns:
        match = pattern.search(url)
        if match:
            year = int(match.group("year"))
            month = int(match.group("month"))
            if 1 <= month <= 12:
                return year, month
    return None


def _extract_md_period_from_name(filename: str) -> tuple[int, int] | None:
    match = MDFILE_PATTERN.search(filename)
    if not match:
        return None

    if match.group(1) is not None:
        return int(match.group(1)), int(match.group(2))
    return int(match.group(3)), int(match.group(4))


def _discover_existing_qd_periods(qd_dir: Path) -> set[tuple[int, int]]:
    periods: set[tuple[int, int]] = set()
    if not qd_dir.exists():
        return periods

    for path in qd_dir.glob("*.csv"):
        match = QDFILE_PATTERN.search(path.name)
        if not match:
            continue
        periods.add((int(match.group(1)), int(match.group(2))))
    return periods


def _discover_existing_md_periods(md_dir: Path) -> set[tuple[int, int]]:
    periods: set[tuple[int, int]] = set()
    if not md_dir.exists():
        return periods

    for path in md_dir.glob("*.csv"):
        ym = _extract_md_period_from_name(path.name)
        if ym is None:
            continue
        periods.add(ym)
    return periods


def _scrape_urls(
    *,
    page_url: str,
    url_pattern: re.Pattern[str],
    url_period_patterns: tuple[re.Pattern[str], ...],
    label: str,
) -> list[str]:
    print(f"Scraping {label} links from: {page_url}")
    with urllib.request.urlopen(page_url) as response:
        raw_html = response.read().decode("utf-8", errors="replace")

    urls: set[str] = set()
    for match in url_pattern.finditer(raw_html):
        href = html.unescape(match.group("href"))
        abs_url = urllib.parse.urljoin(page_url, href)
        # Skip generic current.csv label if it is not month-identifiable.
        if _extract_vintage_year_month(abs_url, patterns=url_period_patterns) is None:
            continue
        urls.add(abs_url)

    return sorted(urls)


def _scrape_monthly_urls(page_url: str) -> list[str]:
    return _scrape_urls(
        page_url=page_url,
        url_pattern=MONTHLY_URL_PATTERN,
        url_period_patterns=MONTHLY_URL_PERIOD_PATTERNS,
        label="monthly",
    )


def _scrape_quarterly_urls(page_url: str) -> list[str]:
    return _scrape_urls(
        page_url=page_url,
        url_pattern=QUARTERLY_URL_PATTERN,
        url_period_patterns=QUARTERLY_URL_PERIOD_PATTERNS,
        label="quarterly",
    )


def _download_monthly_vintages(
    *,
    page_url: str,
    md_dir: Path,
    force: bool,
) -> tuple[int, int]:
    md_dir.mkdir(parents=True, exist_ok=True)
    urls = _scrape_monthly_urls(page_url=page_url)
    if not urls:
        print("No monthly vintage URLs discovered on page.")
        return 0, 0

    existing = _discover_existing_md_periods(md_dir=md_dir)
    downloaded = 0
    skipped = 0

    ordered: list[tuple[tuple[int, int], str]] = []
    for url in urls:
        ym = _extract_vintage_year_month(url, patterns=MONTHLY_URL_PERIOD_PATTERNS)
        if ym is None:
            continue
        ordered.append((ym, url))
    ordered.sort(key=lambda x: x[0])

    for (year, month), url in ordered:
        out_name = _normalize_md_filename(year=year, month=month)
        out_path = md_dir / out_name
        period_key = (year, month)

        if period_key in existing and out_path.exists() and not force:
            skipped += 1
            continue
        if period_key in existing and not force:
            skipped += 1
            continue

        try:
            _download(url=url, destination=out_path, force=force)
            existing.add(period_key)
            downloaded += 1
        except Exception as err:
            print(f"Warning: failed monthly download for {url}: {err}")

    return downloaded, skipped


def _download_quarterly_vintages(
    *,
    page_url: str,
    qd_dir: Path,
    force: bool,
) -> tuple[int, int]:
    qd_dir.mkdir(parents=True, exist_ok=True)
    urls = _scrape_quarterly_urls(page_url=page_url)
    if not urls:
        print("No quarterly vintage URLs discovered on page.")
        return 0, 0

    existing = _discover_existing_qd_periods(qd_dir=qd_dir)
    downloaded = 0
    skipped = 0

    # Download in chronological order.
    ordered: list[tuple[tuple[int, int], str]] = []
    for url in urls:
        ym = _extract_vintage_year_month(url, patterns=QUARTERLY_URL_PERIOD_PATTERNS)
        if ym is None:
            continue
        ordered.append((ym, url))
    ordered.sort(key=lambda x: x[0])

    for (year, month), url in ordered:
        out_name = _normalize_qd_filename(year=year, month=month)
        out_path = qd_dir / out_name
        period_key = (year, month)

        if period_key in existing and out_path.exists() and not force:
            skipped += 1
            continue
        if period_key in existing and not force:
            skipped += 1
            continue

        try:
            _download(url=url, destination=out_path, force=force)
            existing.add(period_key)
            downloaded += 1
        except Exception as err:
            print(f"Warning: failed quarterly download for {url}: {err}")

    return downloaded, skipped


def _consolidate_md_archive_csvs(
    *,
    md_root: Path,
    canonical_md_dir: Path,
    force: bool,
) -> tuple[int, int]:
    """Copy historical MD CSVs from extracted archive folders into canonical MD directory."""
    canonical_md_dir.mkdir(parents=True, exist_ok=True)
    imported = 0
    skipped = 0

    source_files = sorted(p for p in md_root.rglob("*.csv") if not p.resolve().is_relative_to(canonical_md_dir.resolve()))
    for src in source_files:
        ym = _extract_md_period_from_name(src.name)
        if ym is None:
            continue
        year, month = ym
        dst = canonical_md_dir / _normalize_md_filename(year=year, month=month)

        if dst.exists() and not force:
            skipped += 1
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        imported += 1

    return imported, skipped


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    canonical_md_dir = Path(args.monthly_output_dir).expanduser().resolve()
    canonical_qd_dir = Path(args.quarterly_output_dir).expanduser().resolve()
    downloads_dir = (
        Path(args.downloads_dir).expanduser().resolve()
        if args.downloads_dir is not None
        else (output_dir / "downloads")
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "md").mkdir(parents=True, exist_ok=True)
    (output_dir / "qd").mkdir(parents=True, exist_ok=True)
    (output_dir / "archives").mkdir(parents=True, exist_ok=True)
    canonical_md_dir.mkdir(parents=True, exist_ok=True)
    canonical_qd_dir.mkdir(parents=True, exist_ok=True)
    downloads_dir.mkdir(parents=True, exist_ok=True)

    all_extracted: list[Path] = []
    if not args.skip_archives:
        for url in FRED_VINTAGE_URLS:
            filename = _filename_from_url(url)
            zip_path = downloads_dir / filename
            _download(url=url, destination=zip_path, force=args.force)
            extract_root = _archive_extract_root(output_dir=output_dir, url=url)
            extracted = _extract(zip_path=zip_path, output_dir=extract_root, force=args.force)
            all_extracted.extend(extracted)

    md_imported = 0
    md_import_skipped = 0
    for md_source in [output_dir / "md", output_dir / "archives" / "md"]:
        imported, skipped = _consolidate_md_archive_csvs(
            md_root=md_source,
            canonical_md_dir=canonical_md_dir,
            force=args.force,
        )
        md_imported += imported
        md_import_skipped += skipped

    monthly_downloaded = 0
    monthly_skipped = 0
    if not args.skip_monthly:
        monthly_downloaded, monthly_skipped = _download_monthly_vintages(
            page_url=args.fred_page_url,
            md_dir=canonical_md_dir,
            force=args.force,
        )

    quarterly_downloaded = 0
    quarterly_skipped = 0
    if not args.skip_quarterly:
        quarterly_downloaded, quarterly_skipped = _download_quarterly_vintages(
            page_url=args.fred_page_url,
            qd_dir=canonical_qd_dir,
            force=args.force,
        )

    print("Completed historical vintage sync.")
    if all_extracted:
        print("Extracted folders:")
        for path in sorted({p.resolve() for p in all_extracted}):
            print(f" - {path}")
    print(
        "MD archive consolidation: "
        f"imported={md_imported}, skipped_existing={md_import_skipped}, "
        f"target_dir={canonical_md_dir}"
    )
    if not args.skip_monthly:
        print(
            "Monthly vintages: "
            f"downloaded={monthly_downloaded}, skipped_existing={monthly_skipped}, "
            f"target_dir={canonical_md_dir}"
        )
    if not args.skip_quarterly:
        print(
            "Quarterly vintages: "
            f"downloaded={quarterly_downloaded}, skipped_existing={quarterly_skipped}, "
            f"target_dir={canonical_qd_dir}"
        )


if __name__ == "__main__":
    main()

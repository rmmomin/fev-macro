# Captured ALFRED fixtures

JSON records contain sanitized request parameters, exact decoded official API responses, and UTC retrieval times from the 2026-09-04 audit. They are actual responses, not synthetic revisions. Ordinary tests never refresh them. Run `scripts/capture_alfred_fixtures.py` to deliberately refresh the observation fixtures; inspect changes before accepting a corrected archive value.

- GDPC1: quarterly real GDP, history available during 2018–April 2020; observation history starts 2005. One-day queries demonstrate clipped intervals and absence before the 2019Q1 advance release.
- GDP: quarterly nominal GDP, 2019Q1 advance 21062.691 on April 26, second 21048.839 on May 30, third 21060.062 on June 27.
- UNRATE: monthly unemployment. January 2019 first available February 1 at 4.0. [BLS release](https://www.bls.gov/news.release/archives/empsit_02012019.htm)
- CPIAUCSL: monthly seasonally adjusted CPI. January 2019 first available February 13 at 252.673. That release was itself reissued the same day, illustrating why a date cannot certify intraday ordering. [BLS release](https://www.bls.gov/news.release/archives/cpi_02132019.htm)
- Supplemental GDPC1 output-type-3 and vintage-date responses show changed-row keys and the three 2019Q1 revision dates. These are diagnostic fixtures, not the ingestion format.

`release_calendar.csv` has sourced BEA stage labels and published rounded SAAR values. `origins.csv` fixes the 25th of April/July/October/January for 2019Q1–Q4, independent of the eventually realized release day. `series_specs.json` fixes the GDP/UNRATE identity, frequencies and UNRATE level transformation. All choices are a retrospective test specification, not evidence of historical preregistration.

For April 26 snapshots ALFRED reports both numerator and denominator with April 26 bounds, although the preceding quarter was originally released earlier. Bounds clipped by a query must never be mistaken for initial release dates.

"""Retrieve the macroeconomic panels and record their provenance.

Downloads every configured FRED series to the macro cache, records a SHA-256
for each in the provenance log, and reports coverage against the origination
window of the analysis sample. Series already cached are skipped unless
--refresh is given, so re-running is cheap and does not silently change the
data a result was computed from.

A series configured with source "local" is not downloaded; it is expected to
already exist under the cache directory, having been built by another script or
committed to the repository. The mechanism is currently unused - all twelve
series come from FRED - but it is retained so that the SARB repo rate can be
substituted for the interbank call rate without a code change. See
scripts/build_repo_rate.py.

Run:
    python -m scripts.fetch_macro
    python -m scripts.fetch_macro --refresh          re-download everything
    python -m scripts.fetch_macro --panel za         one panel only
"""

from __future__ import annotations

import argparse
import urllib.error
import urllib.request
from pathlib import Path


from src.data.macro import read_fred_csv, series_specs
from src.data.provenance import log_retrieval
from src.utils.config import load_config, resolve_path

USER_AGENT = "pd-ecl-research/0.1 (postgraduate research; contact via repository)"
TIMEOUT_SECONDS = 60

LOCAL_TEMPLATE = """
A series configured with source "local" was not found at

    {path}

Local series are not downloaded. Either build it with the script that produces
it, or change the series to source: FRED in config.yaml and give it a series_id.
""".strip()


def download(url: str, destination: Path) -> int:
    """Fetch a URL to a file, returning the number of bytes written.

    Args:
        url: The URL to fetch.
        destination: Where to write. Parent directories are created.

    Returns:
        Bytes written.

    Raises:
        RuntimeError: If the server refuses the request or the response is
            empty, which for FRED usually means an unknown series id.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            payload = response.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"{exc.code} {exc.reason} for {url}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"could not reach {url}: {exc.reason}") from exc

    if not payload.strip():
        raise RuntimeError(f"empty response from {url}; check the series id")

    destination.write_bytes(payload)
    return len(payload)


def describe(path: Path, name: str) -> str:
    """Summarise a cached series: row count and reference-date range."""
    frame = read_fred_csv(path, name)
    observed = frame.dropna(subset=[name])
    if observed.empty:
        return "no usable observations"
    return (
        f"{len(observed):,} obs, "
        f"{observed['reference_date'].min():%Y-%m} to "
        f"{observed['reference_date'].max():%Y-%m}"
    )


def main() -> None:
    """Fetch every configured series and report what was retrieved."""
    parser = argparse.ArgumentParser(description="Retrieve macroeconomic panels")
    parser.add_argument("--refresh", action="store_true", help="Re-download cached series")
    parser.add_argument(
        "--panel",
        choices=["za", "us"],
        default=None,
        help="Fetch one panel only. Default is both.",
    )
    args = parser.parse_args()

    config = load_config()
    cache = resolve_path(config["macro"]["cache_dir"])
    cache.mkdir(parents=True, exist_ok=True)
    provenance = resolve_path(config["paths"]["provenance_log"])

    panels = [args.panel] if args.panel else ["za", "us"]
    missing_local: list[str] = []
    failures: list[str] = []

    print("=" * 78)
    print("Macroeconomic panel retrieval")
    print("=" * 78)
    print(f"  Cache directory : {cache.relative_to(resolve_path('.'))}")
    print(f"  Publication lag : {'applied' if config['macro']['apply_publication_lag'] else 'NOT applied'}")
    print()

    for panel in panels:
        print(f"{panel.upper()} panel")
        print("-" * 78)

        for spec in series_specs(config, panel):
            path = cache / spec.cache_filename

            if spec.source == "local":
                if not path.is_file():
                    print(f"  {spec.name:<20} MISSING - see note below")
                    missing_local.append(spec.name)
                else:
                    print(f"  {spec.name:<20} local      {describe(path, spec.name)}")
                continue

            if path.is_file() and not args.refresh:
                print(f"  {spec.name:<20} cached     {describe(path, spec.name)}")
                continue

            try:
                written = download(spec.url, path)
            except RuntimeError as exc:
                print(f"  {spec.name:<20} FAILED     {exc}")
                failures.append(f"{spec.name} ({spec.series_id})")
                continue

            log_retrieval(
                path=path,
                source_url=spec.url,
                licence="Public domain (US Federal Reserve Bank of St. Louis, FRED)",
                log_path=provenance,
                notes=(
                    f"panel={panel}; series_id={spec.series_id}; "
                    f"frequency={spec.frequency}; transform={spec.transform}; "
                    f"publication_lag_months={spec.publication_lag_months}"
                ),
            )
            print(
                f"  {spec.name:<20} retrieved  {written:>8,} bytes   "
                f"{describe(path, spec.name)}"
            )

        print()

    print("=" * 78)
    if failures:
        print(f"  {len(failures)} series failed: {', '.join(failures)}")
    if missing_local:
        print(f"  {len(missing_local)} local series missing: {', '.join(missing_local)}")
        print()
        print(LOCAL_TEMPLATE.format(path=cache))
    if not failures and not missing_local:
        print("  All series retrieved. Provenance recorded.")
    print("=" * 78)


if __name__ == "__main__":
    main()

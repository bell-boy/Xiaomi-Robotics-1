#!/usr/bin/env python3
"""Fetch the RoboCasa asset registry in parallel and extract it safely.

RoboCasa's own downloader streams one archive at a time, which capped setup at
one connection's throughput. This keeps the same registry but pulls each archive
over parallel byte ranges, resumes per block, verifies the ZIP CRC before
extracting, and rejects paths that escape the destination.

Run with the simulator environment, after RoboCasa is installed:

    /workspace/robocasa365-eval-venv/bin/python scripts/download_robocasa365_assets.py
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import download_parallel as dp  # noqa: E402  local helper beside this script


def extract(archive, destination):
    """Extract a verified archive, refusing entries that escape the destination."""
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as handle:
        for name in handle.namelist():
            if not (destination/name).resolve().is_relative_to(destination):
                raise ValueError(f'Unsafe ZIP path: {name}')
        corrupt = handle.testzip()
        if corrupt:
            raise RuntimeError(f'Corrupt ZIP entry: {corrupt}')
        handle.extractall(destination)


def fetch_asset(key, config, root, progress=print, keep_blocks=False):
    """Download, verify and extract one registry entry."""
    folder = Path(root)/key
    marker = folder/'DONE'
    if marker.exists():
        return json.loads(marker.read_text())
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder/'assets.zip'
    report = dp.download(config['url'], archive, progress=progress, keep_blocks=keep_blocks)
    destination = Path(config['folder']).parent.resolve()
    extract(archive, destination)
    result = dict(asset=key, url=config['url'], bytes=report['bytes'], sha256=report['sha256'],
                  destination=str(destination), crc_verified=True)
    marker.write_text(json.dumps(result))
    if not keep_blocks:
        archive.unlink()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, default=Path('/workspace/asset-downloads'),
                   help='Working directory; one subdirectory per archive')
    p.add_argument('--concurrency', type=int, default=6, help='Archives fetched at once')
    p.add_argument('--keep-blocks', action='store_true', help='Keep block files and archives')
    args = p.parse_args()
    from robocasa.scripts.download_kitchen_assets import DOWNLOAD_ASSET_REGISTRY
    args.root.mkdir(parents=True, exist_ok=True)
    results = []
    with ThreadPoolExecutor(args.concurrency) as pool:
        futures = [pool.submit(fetch_asset, key, config, args.root, print, args.keep_blocks)
                   for key, config in DOWNLOAD_ASSET_REGISTRY.items()]
        for future in as_completed(futures):
            results.append(future.result())
    (args.root/'DONE').write_text(json.dumps(results, indent=2))
    print(f'Extracted {len(results)} verified asset archives')


if __name__ == '__main__':
    main()

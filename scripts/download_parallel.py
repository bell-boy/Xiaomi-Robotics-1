#!/usr/bin/env python3
"""Resumable range-parallel HTTP downloads with checksum verification.

Single-connection transfers are usually capped well below a rented host's link
speed: the RoboCasa kitchen archives arrived at roughly 30 MB/s per stream even
on a 13 Gb/s machine. Splitting each file into byte ranges and fetching them
concurrently gets past that per-connection limit, and every completed block is
kept on disk so a retry resumes instead of restarting.

    python scripts/download_parallel.py URL OUTPUT [--sha256 HEX] [--size N]

`--manifest files.json` downloads several files at once, for example the two
checkpoints and their shards:

    {"files": [{"url": "...", "output": "/workspace/x.pt", "sha256": "...",
                "size": 10226684862}]}
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

BLOCK = 16*1024*1024
RETRIES = 8


def probe(url, timeout=90):
    """Return the resolved URL and total size using a single-byte range request."""
    request = urllib.request.Request(url, headers={'Range': 'bytes=0-0'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 206:
            raise RuntimeError(f'Server does not support byte ranges: {url}')
        return response.geturl(), int(response.headers['Content-Range'].split('/')[-1])


def plan_blocks(size, block=BLOCK):
    """Byte ranges covering the file, in order."""
    return [(start, min(start+block, size)-1) for start in range(0, size, block)]


def threads_for(size, requested=None):
    """Concurrency for one file: more streams for the large archives."""
    if requested:
        return requested
    return 32 if size > 3_000_000_000 else 8


def sha256(path, chunk=8*1024*1024):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for data in iter(lambda: handle.read(chunk), b''):
            digest.update(data)
    return digest.hexdigest()


def fetch_block(url, start, end, size, target, retries=RETRIES, progress=None):
    """Download one range, reuse an existing complete block, and retry with backoff."""
    expected = end-start+1
    if target.exists() and target.stat().st_size == expected:
        return
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}'})
            with urllib.request.urlopen(request, timeout=180) as response:
                if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{size}':
                    raise RuntimeError('Invalid range response')
                temporary = target.with_suffix('.tmp')
                with temporary.open('wb') as out:
                    remaining = expected
                    while remaining:
                        data = response.read(min(1024*1024, remaining))
                        if not data:
                            raise RuntimeError('Truncated block')
                        out.write(data)
                        remaining -= len(data)
                os.replace(temporary, target)
            return
        except Exception as exc:
            if progress:
                progress(dict(block=str(target), retry=attempt, error=type(exc).__name__))
            if attempt == retries-1:
                raise
            time.sleep(min(2**attempt, 30))


def assemble(blocks, output):
    """Concatenate block files into the final artifact, in range order."""
    with Path(output).open('wb') as out:
        for block in blocks:
            with block.open('rb') as source:
                for data in iter(lambda: source.read(8*1024*1024), b''):
                    out.write(data)


def download(url, output, size=None, sha256_hex=None, threads=None, block=BLOCK,
             work_dir=None, progress=print, keep_blocks=False):
    """Fetch one URL into `output`, returning a report dictionary."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    resolved, actual_size = probe(url)
    if size and size != actual_size:
        raise RuntimeError(f'Size mismatch for {url}: expected {size}, server reports {actual_size}')
    size = actual_size
    work_dir = Path(work_dir or output.parent/(output.name+'.blocks'))
    work_dir.mkdir(parents=True, exist_ok=True)
    ranges = plan_blocks(size, block)
    targets = [work_dir/f'{start:012d}.part' for start, _ in ranges]
    started = time.monotonic()
    finished = 0
    with ThreadPoolExecutor(threads_for(size, threads)) as pool:
        futures = [pool.submit(fetch_block, resolved, start, end, size, target, RETRIES, progress)
                   for (start, end), target in zip(ranges, targets)]
        for future in as_completed(futures):
            future.result()
            finished += 1
            if progress and finished % 25 == 0:
                progress(dict(file=str(output), blocks=finished, total=len(ranges)))
    assemble(targets, output)
    digest = sha256(output)
    if sha256_hex and digest != sha256_hex:
        raise RuntimeError(f'Checksum mismatch for {output}: {digest} != {sha256_hex}')
    if not keep_blocks:
        for target in targets:
            target.unlink()
        work_dir.rmdir()
    report = dict(url=url, output=str(output), bytes=size, sha256=digest,
                  seconds=round(time.monotonic()-started, 2), threads=threads_for(size, threads))
    if progress:
        progress(report)
    return report


def download_manifest(path, concurrency=3, progress=print, **kwargs):
    """Download every file in a manifest, several at a time."""
    files = json.loads(Path(path).read_text())['files']
    results = []
    with ThreadPoolExecutor(concurrency) as pool:
        futures = [pool.submit(download, item['url'], item['output'], item.get('size'),
                               item.get('sha256'), item.get('threads'), progress=progress, **kwargs)
                   for item in files]
        for future in as_completed(futures):
            results.append(future.result())
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('url', nargs='?', help='Source URL; omit when using --manifest')
    p.add_argument('output', nargs='?', help='Destination path; omit when using --manifest')
    p.add_argument('--manifest', type=Path, help='JSON manifest with a "files" list')
    p.add_argument('--sha256', dest='sha256_hex', help='Expected digest; verified after assembly')
    p.add_argument('--size', type=int, help='Expected byte count; verified against the server')
    p.add_argument('--threads', type=int, help='Concurrent ranges per file')
    p.add_argument('--block-mb', type=int, default=16, help='Range size in MB (default 16)')
    p.add_argument('--concurrency', type=int, default=3, help='Files at once in manifest mode')
    p.add_argument('--work-dir', type=Path, help='Where to keep partial blocks')
    p.add_argument('--keep-blocks', action='store_true', help='Keep block files after assembling')
    args = p.parse_args()
    if args.manifest:
        results = download_manifest(args.manifest, args.concurrency,
                                    work_dir=args.work_dir, keep_blocks=args.keep_blocks)
    else:
        if not args.url or not args.output:
            p.error('URL and OUTPUT are required without --manifest')
        results = [download(args.url, args.output, args.size, args.sha256_hex, args.threads,
                            args.block_mb*1024*1024, args.work_dir, keep_blocks=args.keep_blocks)]
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()

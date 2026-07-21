#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Fast parallel downloader for HuggingFace datasets with many small files.

HuggingFace's `snapshot_download` paginates through the entire repo tree via
REST API before downloading. For datasets with 100K+ files (e.g., video datasets
in LeRobot format), this makes downloads extremely slow (~5 files/s).

This script bypasses the tree API entirely by constructing URLs directly from
the known file structure, then downloads files with high concurrency using
asyncio + aiohttp. Achieves 13x+ speedup over snapshot_download.

Usage:
    # Download DROID video dataset (LeRobot format)
    python fast_hf_download.py \
        --repo GEAR-Dreams/DreamZero-DROID-Data \
        --pattern "videos/chunk-{chunk:03d}/{view}/episode_{episode:06d}.mp4" \
        --chunks 0-57 \
        --episodes-per-chunk 1000 \
        --total-episodes 57774 \
        --views "observation.images.exterior_image_1_left,observation.images.exterior_image_2_left,observation.images.wrist_image_left" \
        --output /fsx/datasets/droid_lerobot \
        --concurrency 64

    # Download specific chunk range (resume partial download)
    python fast_hf_download.py \
        --repo GEAR-Dreams/DreamZero-DROID-Data \
        --pattern "videos/chunk-{chunk:03d}/{view}/episode_{episode:06d}.mp4" \
        --chunks 28-57 \
        --episodes-per-chunk 1000 \
        --total-episodes 57774 \
        --views "observation.images.exterior_image_1_left,observation.images.exterior_image_2_left,observation.images.wrist_image_left" \
        --output /fsx/datasets/droid_lerobot \
        --concurrency 64

Why this is faster:
    1. Bypasses the HF tree/metadata API entirely (no pagination through 138K files)
    2. Constructs download URLs directly from known dataset structure
    3. Uses 64 concurrent async downloads (vs ~1-4 sequential in snapshot_download)
    4. HuggingFace CDN does NOT rate-limit file downloads (only the tree API)
    5. Skips already-downloaded files instantly (local existence check)

Requirements:
    pip install aiohttp aiofiles tqdm
"""

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

try:
    import aiohttp
    import aiofiles
    from tqdm import tqdm
except ImportError:
    print("Missing dependencies. Install with:")
    print("  pip install aiohttp aiofiles tqdm")
    sys.exit(1)


RETRY_MAX = 5
RETRY_BASE_DELAY = 1.0  # seconds, exponential backoff
BATCH_SIZE = 1000  # files per asyncio.gather batch (memory management)


async def download_file(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    url: str,
    dest: Path,
    pbar: tqdm,
    stats: dict,
) -> bool:
    """Download a single file with exponential backoff retry."""
    if dest.exists() and dest.stat().st_size > 0:
        stats["skipped"] += 1
        pbar.update(1)
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dest = dest.with_suffix(dest.suffix + ".tmp")

    for attempt in range(RETRY_MAX):
        try:
            async with sem:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        size = 0
                        async with aiofiles.open(tmp_dest, "wb") as f:
                            async for chunk in resp.content.iter_chunked(65536):
                                await f.write(chunk)
                                size += len(chunk)
                        # Atomic rename to avoid partial files
                        tmp_dest.rename(dest)
                        stats["downloaded"] += 1
                        stats["bytes"] += size
                        pbar.update(1)
                        return True
                    elif resp.status == 404:
                        # File doesn't exist (e.g., last chunk has fewer episodes)
                        stats["not_found"] += 1
                        pbar.update(1)
                        return True
                    elif resp.status == 429:
                        wait = RETRY_BASE_DELAY * (2**attempt)
                        stats["retries"] += 1
                        await asyncio.sleep(wait)
                    else:
                        stats["retries"] += 1
                        await asyncio.sleep(RETRY_BASE_DELAY * (attempt + 1))
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            stats["retries"] += 1
            await asyncio.sleep(RETRY_BASE_DELAY * (attempt + 1))
            if attempt == RETRY_MAX - 1:
                print(f"\nFAILED after {RETRY_MAX} attempts: {url} ({e})")

    # Clean up temp file on failure
    if tmp_dest.exists():
        tmp_dest.unlink()
    stats["failed"] += 1
    return False


def build_file_list(
    repo: str,
    pattern: str,
    chunks: tuple,
    episodes_per_chunk: int,
    total_episodes: int,
    views: list,
    output_dir: Path,
    revision: str = "main",
) -> list:
    """Build list of (url, dest_path) tuples from the pattern template.

    Pattern placeholders:
        {chunk}   - chunk index (formatted per pattern, e.g., {chunk:03d})
        {view}    - view name from --views list
        {episode} - episode index (formatted per pattern, e.g., {episode:06d})
    """
    base_url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}"
    start_chunk, end_chunk = chunks
    files = []

    for chunk_idx in range(start_chunk, end_chunk + 1):
        ep_start = chunk_idx * episodes_per_chunk
        ep_end = min(ep_start + episodes_per_chunk, total_episodes)

        for ep in range(ep_start, ep_end):
            for view in views:
                rel_path = pattern.format(chunk=chunk_idx, view=view, episode=ep)
                url = f"{base_url}/{rel_path}"
                dest = output_dir / rel_path
                files.append((url, dest))

    return files


async def run_download(
    files: list,
    token: str,
    concurrency: int,
    skip_existing: bool = True,
):
    """Execute parallel download of all files."""
    # Filter already-downloaded files
    if skip_existing:
        pending = [(url, dest) for url, dest in files if not (dest.exists() and dest.stat().st_size > 0)]
    else:
        pending = files

    total = len(files)
    already_done = total - len(pending)

    print(f"Total files:        {total:,}")
    print(f"Already downloaded: {already_done:,}")
    print(f"Remaining:          {len(pending):,}")
    print(f"Concurrency:        {concurrency}")
    print(flush=True)

    if not pending:
        print("\nAll files already downloaded!")
        return

    stats = {"downloaded": 0, "skipped": 0, "failed": 0, "retries": 0, "not_found": 0, "bytes": 0}
    start_time = time.time()

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    timeout = aiohttp.ClientTimeout(total=300, connect=30)
    connector = aiohttp.TCPConnector(limit=concurrency, ttl_dns_cache=300)

    async with aiohttp.ClientSession(
        headers=headers, timeout=timeout, connector=connector
    ) as session:
        sem = asyncio.Semaphore(concurrency)
        pbar = tqdm(total=len(pending), desc="Downloading", unit="file")

        # Process in batches to limit memory usage
        for i in range(0, len(pending), BATCH_SIZE):
            batch = pending[i : i + BATCH_SIZE]
            await asyncio.gather(
                *[download_file(session, sem, url, dest, pbar, stats) for url, dest in batch]
            )

        pbar.close()

    elapsed = time.time() - start_time
    speed = stats["downloaded"] / elapsed if elapsed > 0 else 0
    mb = stats["bytes"] / (1024 * 1024)

    print(f"\n{'=' * 60}")
    print(f"Download complete in {elapsed:.1f}s")
    print(f"  Downloaded: {stats['downloaded']:,} files ({mb:.1f} MB)")
    print(f"  Skipped:    {stats['skipped']:,} (already existed)")
    print(f"  Not found:  {stats['not_found']:,} (404)")
    print(f"  Failed:     {stats['failed']:,}")
    print(f"  Retries:    {stats['retries']:,}")
    print(f"  Speed:      {speed:.1f} files/s ({mb/elapsed*8:.1f} Mbps)")
    print(f"{'=' * 60}")

    if stats["failed"] > 0:
        print(f"\nWARNING: {stats['failed']} files failed to download.")
        sys.exit(1)


def parse_chunk_range(s: str) -> tuple:
    """Parse chunk range like '0-57' or '28-57'."""
    parts = s.split("-")
    if len(parts) == 2:
        return int(parts[0]), int(parts[1])
    elif len(parts) == 1:
        return int(parts[0]), int(parts[0])
    else:
        raise ValueError(f"Invalid chunk range: {s}")


def main():
    parser = argparse.ArgumentParser(
        description="Fast parallel downloader for HuggingFace datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="HuggingFace dataset repo (e.g., GEAR-Dreams/DreamZero-DROID-Data)",
    )
    parser.add_argument(
        "--pattern",
        required=True,
        help='File path pattern with {chunk}, {view}, {episode} placeholders '
        '(e.g., "videos/chunk-{chunk:03d}/{view}/episode_{episode:06d}.mp4")',
    )
    parser.add_argument(
        "--chunks",
        required=True,
        help="Chunk range to download (e.g., 0-57 or 28-57)",
    )
    parser.add_argument(
        "--episodes-per-chunk",
        type=int,
        required=True,
        help="Number of episodes per chunk (e.g., 1000)",
    )
    parser.add_argument(
        "--total-episodes",
        type=int,
        required=True,
        help="Total episodes in dataset (e.g., 57774). Last chunk may have fewer.",
    )
    parser.add_argument(
        "--views",
        required=True,
        help="Comma-separated view/camera names",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory (dataset root)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=64,
        help="Number of parallel downloads (default: 64)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HuggingFace token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Git revision/branch (default: main)",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-download files even if they exist locally",
    )

    args = parser.parse_args()

    # Resolve token
    token = args.token or os.environ.get("HF_TOKEN", "")

    # Parse inputs
    chunk_range = parse_chunk_range(args.chunks)
    views = [v.strip() for v in args.views.split(",")]
    output_dir = Path(args.output)

    print(f"Repository:    {args.repo}")
    print(f"Pattern:       {args.pattern}")
    print(f"Chunks:        {chunk_range[0]}-{chunk_range[1]}")
    print(f"Episodes/chunk:{args.episodes_per_chunk}")
    print(f"Total episodes:{args.total_episodes}")
    print(f"Views:         {views}")
    print(f"Output:        {output_dir}")
    print(f"Revision:      {args.revision}")
    print()

    # Build file list
    files = build_file_list(
        repo=args.repo,
        pattern=args.pattern,
        chunks=chunk_range,
        episodes_per_chunk=args.episodes_per_chunk,
        total_episodes=args.total_episodes,
        views=views,
        output_dir=output_dir,
        revision=args.revision,
    )

    # Run download
    asyncio.run(run_download(
        files=files,
        token=token,
        concurrency=args.concurrency,
        skip_existing=not args.no_skip_existing,
    ))


if __name__ == "__main__":
    main()

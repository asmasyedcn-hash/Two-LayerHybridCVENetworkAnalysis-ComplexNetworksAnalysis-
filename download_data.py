"""
download_data.py
================
MITS 6700G Group 8 — Data Acquisition Script

Downloads real NVD CVE 2021 data from:
  fkie-cad/nvd-json-data-feeds (Fraunhofer FKIE)
  https://github.com/fkie-cad/nvd-json-data-feeds

Data source: NVD API 2.0 (NIST) mirrored by Fraunhofer FKIE.
The CVE records (IDs, CVSS scores, CPEs) are identical to
official NIST data. NIST legacy JSON feeds were retired Dec 2023.

RUN THIS FIRST:
    python download_data.py

THEN RUN:
    python main.py
"""

import os
import sys
import lzma
import shutil
import hashlib

# ── Try to import requests, guide user if missing ─────────────────
try:
    import requests
except ImportError:
    print("[DOWNLOAD] requests not installed.")
    print("[DOWNLOAD] Run: pip install requests")
    sys.exit(1)

# ── Config — DO NOT CHANGE ─────────────────────────────────────────
YEAR          = "2021"
FILENAME_XZ   = f"CVE-{YEAR}.json.xz"
FILENAME_JSON = f"CVE-{YEAR}.json"
OUTPUT_DIR    = "data"
OUTPUT_FILE   = os.path.join(OUTPUT_DIR, "nvdcve-1.1-2021.json")

# fkie-cad static release URL — always points to latest build
DOWNLOAD_URL  = (
    f"https://github.com/fkie-cad/nvd-json-data-feeds"
    f"/releases/latest/download/{FILENAME_XZ}"
)
META_URL      = (
    f"https://github.com/fkie-cad/nvd-json-data-feeds"
    f"/releases/latest/download/CVE-{YEAR}.meta"
)

# ── Helper: download with progress ────────────────────────────────
def download_file(url, dest_path):
    """Download a file with progress reporting."""
    print(f"[DOWNLOAD] Connecting to: {url}")
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        print("[DOWNLOAD] ✗ Cannot connect. Check your internet connection.")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"[DOWNLOAD] ✗ HTTP error: {e}")
        sys.exit(1)

    total_bytes = int(response.headers.get('content-length', 0))
    downloaded  = 0
    chunk_size  = 1024 * 1024  # 1 MB chunks

    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_bytes > 0:
                    pct = downloaded / total_bytes * 100
                    mb  = downloaded / (1024 * 1024)
                    total_mb = total_bytes / (1024 * 1024)
                    print(f"[DOWNLOAD]   {mb:.1f} MB / {total_mb:.1f} MB"
                          f"  ({pct:.0f}%)", end='\r')

    print()  # newline after progress
    return downloaded

# ── Helper: verify sha256 against meta file ───────────────────────
def verify_checksum(json_path, meta_url):
    """Download meta file and verify sha256 of the JSON."""
    print("[DOWNLOAD] Fetching checksum from meta file...")
    try:
        r = requests.get(meta_url, timeout=30)
        r.raise_for_status()
        expected_sha = None
        for line in r.text.splitlines():
            if line.startswith("sha256:"):
                expected_sha = line.split(":")[1].strip()
                break
        if not expected_sha:
            print("[DOWNLOAD] ⚠ Could not parse sha256 from meta — skipping verify")
            return True

        print("[DOWNLOAD] Computing sha256 of downloaded file...")
        sha256 = hashlib.sha256()
        with open(json_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        actual_sha = sha256.hexdigest()

        if actual_sha == expected_sha:
            print(f"[DOWNLOAD] ✓ Checksum verified: {actual_sha[:16]}...")
            return True
        else:
            print(f"[DOWNLOAD] ✗ Checksum MISMATCH!")
            print(f"[DOWNLOAD]   Expected : {expected_sha}")
            print(f"[DOWNLOAD]   Got      : {actual_sha}")
            return False
    except Exception as e:
        print(f"[DOWNLOAD] ⚠ Could not verify checksum: {e}")
        return True  # non-fatal — continue anyway

# ── Helper: decompress .xz file ───────────────────────────────────
def decompress_xz(xz_path, json_path):
    """Decompress .xz file to .json using Python's built-in lzma."""
    print(f"[DOWNLOAD] Decompressing {xz_path} ...")
    try:
        with lzma.open(xz_path, 'rb') as f_in:
            with open(json_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        size_mb = os.path.getsize(json_path) / (1024 * 1024)
        print(f"[DOWNLOAD] ✓ Decompressed: {json_path} ({size_mb:.1f} MB)")
    except lzma.LZMAError as e:
        print(f"[DOWNLOAD] ✗ Decompression failed: {e}")
        sys.exit(1)

# ── Helper: quick verify Log4Shell present ─────────────────────────
def verify_log4shell_in_file(json_path):
    """Check CVE-2021-44228 exists in the downloaded file."""
    print("[DOWNLOAD] Verifying Log4Shell (CVE-2021-44228) is present...")
    # Read first 5MB only to find Log4Shell quickly without loading all
    with open(json_path, 'r', encoding='utf-8') as f:
        chunk = f.read(5 * 1024 * 1024)
    if "CVE-2021-44228" in chunk:
        print("[DOWNLOAD] ✓ Log4Shell confirmed in dataset")
        return True
    else:
        # Try reading full file if not in first chunk
        with open(json_path, 'r', encoding='utf-8') as f:
            content = f.read()
        if "CVE-2021-44228" in content:
            print("[DOWNLOAD] ✓ Log4Shell confirmed in dataset")
            return True
        else:
            print("[DOWNLOAD] ✗ WARNING: CVE-2021-44228 not found!")
            print("[DOWNLOAD]   Something may have gone wrong with download.")
            return False

# ── Main download flow ─────────────────────────────────────────────
def main():
    """Main entry point for data download."""
    print("=" * 65)
    print("  MITS 6700G — Data Download Script")
    print("  Source: fkie-cad/nvd-json-data-feeds (Fraunhofer FKIE)")
    print("  Data  : NVD CVE 2021 (real NVD data, NVD API 2.0 mirror)")
    print("=" * 65)

    # ── Step 1: Check if already downloaded ───────────────────────
    if os.path.exists(OUTPUT_FILE):
        size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
        print(f"[DOWNLOAD] File already exists: {OUTPUT_FILE} ({size_mb:.1f} MB)")
        print(f"[DOWNLOAD] Delete it and re-run if you want a fresh download.")
        verify_log4shell_in_file(OUTPUT_FILE)
        print(f"\n[DOWNLOAD] ✅ Ready — run: python main.py")
        return

    # ── Step 2: Create data directory ─────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[DOWNLOAD] Output directory: {OUTPUT_DIR}/")

    # ── Step 3: Download compressed .xz file ──────────────────────
    xz_path   = os.path.join(OUTPUT_DIR, FILENAME_XZ)
    json_path = os.path.join(OUTPUT_DIR, FILENAME_JSON)

    print(f"\n[DOWNLOAD] Downloading CVE-{YEAR} data feed...")
    print(f"[DOWNLOAD] URL: {DOWNLOAD_URL}")
    print(f"[DOWNLOAD] This is ~20-50 MB compressed. Please wait...\n")

    bytes_downloaded = download_file(DOWNLOAD_URL, xz_path)
    size_mb = bytes_downloaded / (1024 * 1024)
    print(f"[DOWNLOAD] ✓ Downloaded: {xz_path} ({size_mb:.1f} MB)")

    # ── Step 4: Decompress ────────────────────────────────────────
    print()
    decompress_xz(xz_path, json_path)

    # ── Step 5: Verify checksum ───────────────────────────────────
    print()
    verify_checksum(json_path, META_URL)

    # ── Step 6: Rename to expected filename ───────────────────────
    print(f"\n[DOWNLOAD] Renaming to project filename...")
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
    os.rename(json_path, OUTPUT_FILE)
    final_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"[DOWNLOAD] ✓ Final file: {OUTPUT_FILE} ({final_mb:.1f} MB)")

    # ── Step 7: Clean up .xz ──────────────────────────────────────
    if os.path.exists(xz_path):
        os.remove(xz_path)
        print(f"[DOWNLOAD] ✓ Cleaned up: {xz_path}")

    # ── Step 8: Verify Log4Shell is present ───────────────────────
    print()
    log4_ok = verify_log4shell_in_file(OUTPUT_FILE)

    # ── Done ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    if log4_ok:
        print("  ✅  DOWNLOAD COMPLETE")
        print(f"  📁  File: {OUTPUT_FILE}")
        print(f"  📊  Size: {final_mb:.1f} MB")
        print(f"  🔑  Log4Shell (CVE-2021-44228): CONFIRMED")
        print("  ▶   Next step: python main.py")
    else:
        print("  ⚠   DOWNLOAD COMPLETE WITH WARNING")
        print("      Log4Shell not found — check the file manually")
        print(f"  📁  File: {OUTPUT_FILE}")
    print("=" * 65)
    print()
    print("  DATA SOURCE CITATION FOR YOUR REPORT:")
    print("  Helmke, R. (2024). nvd-json-data-feeds: Community")
    print("  reconstruction of the legacy JSON NVD Data Feeds.")
    print("  Fraunhofer FKIE.")
    print("  https://github.com/fkie-cad/nvd-json-data-feeds")
    print()
    print("  The underlying CVE data originates from:")
    print("  National Vulnerability Database (NVD), NIST.")
    print("  https://nvd.nist.gov")
    print("=" * 65)


if __name__ == "__main__":
    main()

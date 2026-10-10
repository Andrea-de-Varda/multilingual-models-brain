"""
Download the large data files that are distributed as GitHub release assets (audio stimuli and two large
fMRI source files), extract them to their place in the repository, and verify every file against the
SHA-256 checksums in data_manifest.json.

Only the Python standard library is needed. Run from anywhere:
    python download_data.py                    # everything (about 4.3 GB)
    python download_data.py --list             # show what is available
    python download_data.py audio_study1.zip   # a single asset
    python download_data.py --verify-only      # check files that are already in place

If the download is interrupted, run the command again: complete, verified files are skipped.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.request
import zipfile

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(REPO_ROOT, "data_manifest.json")


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def file_ok(entry):
    path = os.path.join(REPO_ROOT, entry["path"])
    return os.path.isfile(path) and os.path.getsize(path) == entry["bytes"] and sha256(path) == entry["sha256"]


def download(url, dest):
    tmp = dest + ".part"
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            block = r.read(1 << 20)
            if not block:
                break
            f.write(block)
            done += len(block)
            if total:
                sys.stdout.write(f"\r  {done / 1e6:8.1f} / {total / 1e6:.1f} MB")
                sys.stdout.flush()
    sys.stdout.write("\n")
    os.replace(tmp, dest)


def fetch_asset(name, spec, base, keep_zip):
    if all(file_ok(e) for e in spec["files"]):
        print(f"[ok] {name}: all {len(spec['files'])} files already present and verified")
        return True
    os.makedirs(os.path.join(REPO_ROOT, ".downloads"), exist_ok=True)
    zpath = os.path.join(REPO_ROOT, ".downloads", name)
    if not (os.path.isfile(zpath) and sha256(zpath) == spec["sha256"]):
        src = f"{base}/{name}"
        print(f"[get] {name} ({spec['bytes'] / 1e6:.0f} MB) from {src}")
        if os.path.isdir(base):  # local mirror (used for testing)
            shutil.copyfile(os.path.join(base, name), zpath)
        else:
            download(src, zpath)
        if sha256(zpath) != spec["sha256"]:
            print(f"[FAIL] checksum of {name} does not match the manifest")
            return False
    with zipfile.ZipFile(zpath) as zf:
        for e in spec["files"]:
            if not file_ok(e):
                zf.extract(e["path"], REPO_ROOT)
    bad = [e["path"] for e in spec["files"] if not file_ok(e)]
    if bad:
        print(f"[FAIL] {name}: {len(bad)} files do not match their checksum: {bad}")
        return False
    if not keep_zip:
        os.remove(zpath)
    print(f"[ok] {name}: {len(spec['files'])} files extracted and verified")
    return True


def main():
    with open(MANIFEST) as f:
        manifest = json.load(f)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("assets", nargs="*", help="asset names (default: all)")
    ap.add_argument("--list", action="store_true", help="list the assets and exit")
    ap.add_argument("--verify-only", action="store_true", help="only verify files already in place")
    ap.add_argument("--keep-zip", action="store_true", help="keep the downloaded archives in .downloads/")
    ap.add_argument("--base-url", default=manifest["base_url"], help="release URL or a local folder with the assets")
    args = ap.parse_args()

    assets = manifest["assets"]
    if args.list:
        for name, a in assets.items():
            print(f"{name:28s} {a['bytes'] / 1e6:8.0f} MB  {len(a['files']):3d} files  {a['description']}")
        return 0
    names = args.assets or list(assets)
    unknown = [n for n in names if n not in assets]
    if unknown:
        print("unknown asset(s):", unknown, "- use --list")
        return 1
    ok = True
    for name in names:
        if args.verify_only:
            bad = [e["path"] for e in assets[name]["files"] if not file_ok(e)]
            print(f"[{'ok' if not bad else 'FAIL'}] {name}: {len(assets[name]['files']) - len(bad)}/{len(assets[name]['files'])} files verified")
            ok &= not bad
        else:
            ok &= fetch_asset(name, assets[name], args.base_url, args.keep_zip)
    print("All requested files verified." if ok else "Some files failed verification (see above).")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

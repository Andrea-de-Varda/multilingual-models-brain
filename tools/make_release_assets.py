"""
Maintainer script: package the large data files that are too big for the git repository into GitHub
release assets, and write the manifest (data_manifest.json) that download_data.py uses to fetch and verify them.

Every original file is recorded with its size and SHA-256 checksum, and every asset (zip archive) is
recorded with its own size and checksum. Archives use ZIP_STORED for audio and parquet files (already
incompressible) and ZIP_DEFLATED for csv files. Each asset stays below GitHub's 2 GiB per-file limit.

Usage (from the repository root):
    python tools/make_release_assets.py --out release_assets
then upload every file in release_assets/ to the GitHub release named in data_manifest.json, e.g.
    gh release create data-v1 release_assets/*.zip --title "Data (audio and large source files)" --notes-file release_assets/RELEASE_NOTES.md
"""
import argparse
import hashlib
import json
import os
import zipfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RELEASE_TAG = "data-v1"
REPO_SLUG = "Andrea-de-Varda/multilingual-models-brain"
MAX_ASSET_BYTES = 2 * 1024**3 - 1  # GitHub limit: each release asset must be under 2 GiB

STUDY1_LANGS = ["Afrikaans", "Dutch", "Farsi", "French", "Lithuanian", "Marathi", "Norwegian",
                "Romanian", "Spanish", "Tamil", "Turkish", "Vietnamese", "English"]
STUDY2_LANGS = ["Arabic", "German", "Hindi", "Italian", "Korean", "Mandarin", "Polish", "Portuguese",
                "Russian", "English"]

ASSETS = {
    "audio_study1.zip": {
        "description": "Study I audio stimuli: the Alice in Wonderland passage in each of the 12 languages analysed (plus the English version, used only for the word-overlap measure of Figure 1C). Data from Malik-Moraleda, Ayyash, et al. (2022).",
        "files": [f"sound_data/{l}/{l}_Intact_Long/{l}.wav" for l in STUDY1_LANGS],
        "compression": zipfile.ZIP_STORED,
    },
    "audio_study2.zip": {
        "description": "Study II audio stimuli: three Alice in Wonderland passages in each of the 9 newly tested languages (plus English, used only for the word-overlap measure of Figure 1C).",
        "files": [f"confirmatory/sound_data/Passage_{p}/{l}.wav" for p in (1, 2, 3) for l in STUDY2_LANGS],
        "compression": zipfile.ZIP_STORED,
    },
    "audio_natstories.zip": {
        "description": "NatStories audio stimuli of the 9 analysed stories (numbering 1-7, 9, 10; story 8 has no fMRI data).",
        "files": [f"additional_analyses/NaturalStories/sound_data/{n}.wav" for n in (1, 2, 3, 4, 5, 6, 7, 9, 10)],
        "compression": zipfile.ZIP_STORED,
    },
    "natstories_timeseries.zip": {
        "description": "NatStories fMRI time series (language fROIs, all participants), input of additional_analyses/NaturalStories/natstor_fmri_timeseries.py.",
        "files": ["additional_analyses/NaturalStories/NaturalStories_TimeSeries_20220615.csv"],
        "compression": zipfile.ZIP_DEFLATED,
    },
    "pereira2018_parquet.zip": {
        "description": "Pereira2018 voxel-level effect sizes in the language fROIs (Experiments 2 and 3), input used to derive additional_analyses/pereira/pereira_averaged.csv.",
        "files": ["additional_analyses/pereira/Pereira.parquet.gzip"],
        "compression": zipfile.ZIP_STORED,
    },
}


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="release_assets", help="output directory (relative to the repository root)")
    args = ap.parse_args()
    out_dir = os.path.join(REPO_ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)

    manifest = {"release_tag": RELEASE_TAG,
                "base_url": f"https://github.com/{REPO_SLUG}/releases/download/{RELEASE_TAG}",
                "assets": {}}
    for asset_name, spec in ASSETS.items():
        files = []
        for rel in spec["files"]:
            src = os.path.join(REPO_ROOT, rel)
            if not os.path.isfile(src):
                raise FileNotFoundError(src)
            files.append({"path": rel, "bytes": os.path.getsize(src), "sha256": sha256(src)})
            print(f"  hashed {rel}")
        asset_path = os.path.join(out_dir, asset_name)
        with zipfile.ZipFile(asset_path, "w", compression=spec["compression"], allowZip64=True) as zf:
            for f in files:
                zf.write(os.path.join(REPO_ROOT, f["path"]), arcname=f["path"])
        size = os.path.getsize(asset_path)
        if size > MAX_ASSET_BYTES:
            raise RuntimeError(f"{asset_name} is {size} bytes, above the 2 GiB GitHub limit; split it")
        manifest["assets"][asset_name] = {"description": spec["description"], "bytes": size,
                                          "sha256": sha256(asset_path), "files": files}
        print(f"{asset_name}: {size / 1e6:.1f} MB, {len(files)} files")

    with open(os.path.join(REPO_ROOT, "data_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
        f.write("\n")
    with open(os.path.join(out_dir, "RELEASE_NOTES.md"), "w") as f:
        f.write("Large data files for the repository (audio stimuli and two large fMRI source files). "
                "Download and verify them with `python download_data.py` from the repository root; "
                "checksums of every file are in `data_manifest.json`.\n\n")
        for name, a in manifest["assets"].items():
            f.write(f"- `{name}` ({a['bytes'] / 1e6:.0f} MB, {len(a['files'])} files): {a['description']}\n")
    print("wrote data_manifest.json and", os.path.join(args.out, "RELEASE_NOTES.md"))


if __name__ == "__main__":
    main()

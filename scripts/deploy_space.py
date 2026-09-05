"""Upload space/index.html (and README.md if changed) to the HF Space.

The Space is static HTML that fetches results live from GitHub raw `main`, so
result YAMLs reach it via the git push; only the page itself needs uploading.
Run with --dry-run first.
"""
import argparse, os, sys, hashlib, pathlib

# Repo root: this file lives in scripts/, the page in space/.
ROOT = pathlib.Path(__file__).resolve().parents[1]
SPACE_ID = "ghananlpcommunity/nsanku-asr-benchmark"


def load_env():
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    load_env()
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))

    for name in ("index.html", "README.md"):
        local = ROOT / "space" / name
        if not local.exists():
            print(f"  {name}: missing locally, skipped")
            continue
        try:
            remote_path = api.hf_hub_download(SPACE_ID, name, repo_type="space")
            same = (hashlib.md5(open(remote_path, "rb").read()).hexdigest()
                    == hashlib.md5(local.read_bytes()).hexdigest())
        except Exception:
            same = False
        if same:
            print(f"  {name}: identical, nothing to upload")
            continue
        print(f"  {name}: differs ({local.stat().st_size} bytes local)")
        if args.dry_run:
            print(f"    [dry-run] would upload to {SPACE_ID}")
            continue
        api.upload_file(path_or_fileobj=str(local), path_in_repo=name,
                        repo_id=SPACE_ID, repo_type="space",
                        commit_message="Add CPU-models track + standardized param labels")
        print(f"    uploaded -> https://huggingface.co/spaces/{SPACE_ID}")


if __name__ == "__main__":
    main()

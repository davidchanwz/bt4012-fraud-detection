"""Download the raw BankSim CSV into data/raw/ and check it's the right file.

Usage: python -m src.data.download
If the mirror ever goes down, download bs140513_032310.csv manually from
https://www.kaggle.com/datasets/ealaxi/banksim1 and put it in data/raw/.
"""
import hashlib
import sys
import urllib.request

from src.config import RAW_DIR, RAW_FILE, RAW_GIT_BLOB_SHA1, RAW_SIZE_BYTES, RAW_URL


def git_blob_sha1(path):
    # same hash `git hash-object` gives, so it can be compared to the GitHub blob
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data).hexdigest()


def verify(path):
    size = path.stat().st_size
    if size != RAW_SIZE_BYTES:
        return False, f"size {size} != expected {RAW_SIZE_BYTES}"
    sha = git_blob_sha1(path)
    if sha != RAW_GIT_BLOB_SHA1:
        return False, f"hash {sha} != expected {RAW_GIT_BLOB_SHA1}"
    return True, "ok"


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_FILE.exists():
        ok, msg = verify(RAW_FILE)
        if ok:
            print(f"{RAW_FILE.name} already there and verified")
            return
        print(f"existing file failed check ({msg}), downloading again")

    print(f"downloading {RAW_URL}")
    urllib.request.urlretrieve(RAW_URL, RAW_FILE)
    ok, msg = verify(RAW_FILE)
    if not ok:
        print(f"downloaded file failed check: {msg}")
        sys.exit(1)
    print(f"saved to {RAW_FILE} ({RAW_SIZE_BYTES / 1e6:.1f} MB), verified")


if __name__ == "__main__":
    main()

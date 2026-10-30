"""Download the pinned CALVADOS2 M1_24 Rg training snapshots."""

import hashlib
from pathlib import Path
from urllib.request import urlopen

COMMIT = "85979b1b4123b6b5391b617d16551969eda9f56e"
BASE_URL = (
    f"https://raw.githubusercontent.com/KULL-Centre/papers/{COMMIT}/"
    "2022/CG-cutoffs-Tesei-et-al/optimization/M1_24"
)
RAW = Path(__file__).resolve().parent / "raw" / "M1_24"
CHECKSUMS = {
    "residues.csv": "e36eeabc8e547c7b8674f6b9490a7f0702cd330f",
    "selHPS.csv": "f354e00f20a72e9d6f4466386054e5bb1fa4c140",
    "1_init_proteinsRgs.pkl": "f34e81d69c2717b15a1f9a30a9f32ada3859e9b5",
    "2_init_proteinsRgs.pkl": "82d373612d8c312c38ef0e1a44d374cc1fea3e36",
    "3_init_proteinsRgs.pkl": "9e401a1e59e4e28803fb1b8d4555fd1aa6f940d4",
    "4_init_proteinsRgs.pkl": "1cee097f6fef6c95f1ada7ab00b1886850d7afb2",
    "5_init_proteinsRgs.pkl": "9ac9518ed243491e043c77e2c26731a55d05ff3c",
    "6_init_proteinsRgs.pkl": "5fae4e7dadc358ca52f6c2567d71da8967da7210",
    "7_init_proteinsRgs.pkl": "4976e6a28b185423c05e5355e1cdb01b1ddbf659",
    "8_init_proteinsRgs.pkl": "9a60adafcc0ee83a82f56534b39be99a72845668",
    "9_init_proteinsRgs.pkl": "b533490f18c462bf20bd3a54d48974ac1aef53d3",
    "10_init_proteinsRgs.pkl": "ac0637c46e8559ccc802e8f353f7ed3941019698",
    "11_init_proteinsRgs.pkl": "63d1092f72f86748e88ce71e10aa071adc6ff682",
}


def verified_bytes(path: Path) -> bytes:
    data = path.read_bytes()
    digest = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    if digest != CHECKSUMS[path.name]:
        raise ValueError(f"CALVADOS checksum mismatch: {path}")
    return data


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name in CHECKSUMS:
        destination = RAW / name
        if destination.exists():
            verified_bytes(destination)
            print(f"Verified {name}")
            continue
        with urlopen(f"{BASE_URL}/{name}", timeout=120) as response:
            data = response.read()
        digest = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if digest != CHECKSUMS[name]:
            raise ValueError(f"CALVADOS checksum mismatch: {name}")
        temporary = destination.with_suffix(destination.suffix + ".part")
        temporary.write_bytes(data)
        temporary.replace(destination)
        print(f"Downloaded {name}")


if __name__ == "__main__":
    main()

"""Download pinned CALVADOS2 M1_24 snapshots and optimization histories."""

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
    "1_chi2.pkl": "23eabbbdda52db158842bbbf7c039087982097d2",
    "2_chi2.pkl": "405c703dcf84bf7b6f9deb7cbebe3fb1c36cacd6",
    "3_chi2.pkl": "187359d586d726141b83df8ecea65d55f69bb841",
    "4_chi2.pkl": "9b398408688b00b80bfec8c03c1b514fa97fc3b3",
    "5_chi2.pkl": "542a3ab9b01d751b566b15f85d60c722fd242574",
    "6_chi2.pkl": "3e75a263b0a4dfc7f42fc0a1c6a4d7eb7a266ddc",
    "7_chi2.pkl": "c84ee207a8a563264891dd5ec3926a4d8cae977b",
    "8_chi2.pkl": "b685499054c52846ea11f8badc792d4443ac2a82",
    "9_chi2.pkl": "fb0377f0e5585556c7a42c8bf06b4aaa43e4be54",
    "10_chi2.pkl": "c2d902124f83cfa6dfb32e995e860a93a34c8c06",
    "11_chi2.pkl": "264afb4e5424a2c844e657fd580a1de412848235",
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

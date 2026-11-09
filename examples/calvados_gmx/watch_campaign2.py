"""Watch the campaign and refit the interim GP every 50 new samples.

Polls ``campaign2/hst_<i>/rg.txt``; every time the count of cached
points crosses a new multiple of 50 (50, 100, 150, 200, 250), runs
``fit_campaign2_interim`` and appends its held-out metrics to
``campaign2/fit_interim/history.jsonl``. Exits when the campaign
process finishes or 250 points are cached.

Run from the repository root alongside the campaign:

    python -m examples.calvados_gmx.watch_campaign2
"""

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
from examples.calvados_gmx.run_campaign2 import N_SAMPLES, OUT  # noqa: E402

HISTORY = OUT / "fit_interim" / "history.jsonl"
DEFUNCT = {"campaign2_run.log"} 
EVERY = 50


def n_cached():
    if not OUT.is_dir():
        return 0
    return sum(
        1 for i in range(N_SAMPLES)
        if (OUT / f"hst_{i:03d}" / "rg.txt").is_file()
    )


def campaign_running():
    output = subprocess.run(
        ["pgrep", "-f", "examples.calvados_gmx.run_campaign2"],
        capture_output=True, text=True,
    )
    return output.returncode == 0


def refit(n):
    result = subprocess.run(
        [sys.executable, "-m", "examples.calvados_gmx.fit_campaign2_interim"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"refit at n={n} FAILED:\n{result.stderr[-2000:]}", flush=True)
        return None
    metrics = json.loads((OUT / "fit_interim" / "metrics.json").read_text())
    record = {"n_samples": n, **metrics}
    with HISTORY.open("a") as handle:
        handle.write(json.dumps(record) + "\n")
    test = record["test_metrics"]
    print(
        f"[n={n}] GP RMSE {test['rmse_nm']:.4f} nm, R2 {test['r2']:+.3f} | "
        f"baseline R2 {record['baseline_train_mean']['r2']:+.3f} | "
        f"signal std {record['signal_std_nm']:.4f} nm",
        flush=True,
    )
    return record


def main():
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    # include already-passed thresholds when starting late
    current = n_cached()
    print(f"watching campaign2: {current} cached, campaign "
          f"{'running' if campaign_running() else 'not running'}", flush=True)
    try:
        while True:
            n = n_cached()
            while done and max(done) + EVERY < n and len(done) < N_SAMPLES // EVERY:
                pass  # guard against skipping thresholds on a jump
            threshold = (n // EVERY) * EVERY
            if threshold > 0 and threshold not in done and threshold % EVERY == 0:
                done.add(threshold)
                refit(threshold)
            if n >= N_SAMPLES:
                print("campaign complete: final refit", flush=True)
                refit(N_SAMPLES)
                return
            if not campaign_running():
                print("campaign process gone; final refit at current count",
                      flush=True)
                refit(n)
                return
            time.sleep(120)
    except KeyboardInterrupt:
        print("watcher stopped", flush=True)


if __name__ == "__main__":
    main()

"""
run_all.py  --  Run XGBoost, LSTM and Hybrid training scripts back to back.

Place this in the same folder as:
    train_xgboost.py
    train_lstm.py
    train_hybrid.py

Each script runs independently and saves results to its own folder:
    results/XGBoost Training/
    results/LSTM Training/
    results/XGB+LSTM Training/

Usage:
    python run_all.py                  # runs all three
    python run_all.py --xgboost        # XGBoost only
    python run_all.py --lstm           # LSTM only
    python run_all.py --hybrid         # Hybrid only
    python run_all.py --xgboost --lstm # any combination
"""

import sys
import os
import time
import argparse
import subprocess


SCRIPTS = {
    "XGBoost"      : "train_xgboost.py",
    "LSTM"         : "train_lstm.py",
    "XGB+LSTM"     : "train_hybrid.py",
}


def run_script(name, script_name):
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_name)
    if not os.path.isfile(script_path):
        print(f"  [ERROR] {script_name} not found in same folder as run_all.py")
        return False, 0.0

    print(f"\n{'='*60}")
    print(f"  Running: {name}  ({script_name})")
    print(f"{'='*60}\n")
    t0 = time.time()

    result = subprocess.run(
        [sys.executable, script_path],
        cwd=os.path.dirname(os.path.abspath(__file__))
    )

    elapsed = time.time() - t0
    success = result.returncode == 0
    status  = "DONE" if success else "FAILED"
    print(f"\n  {name}: {status} in {elapsed:.1f}s")
    return success, elapsed


def main():
    parser = argparse.ArgumentParser(
        description="Run sleep posture training scripts."
    )
    parser.add_argument("--xgboost", action="store_true", help="Run XGBoost only")
    parser.add_argument("--lstm",    action="store_true", help="Run LSTM only")
    parser.add_argument("--hybrid",  action="store_true", help="Run Hybrid only")
    args = parser.parse_args()

    # if no flags given, run all three
    run_any = args.xgboost or args.lstm or args.hybrid
    to_run  = {
        "XGBoost"  : args.xgboost or not run_any,
        "LSTM"     : args.lstm    or not run_any,
        "XGB+LSTM" : args.hybrid  or not run_any,
    }

    print("\n" + "="*60)
    print("  Sleep Posture — Training Runner")
    print("="*60)
    print("  Order: XGBoost → LSTM → XGB+LSTM Hybrid")
    print("  Results saved separately per model folder.")
    print("="*60)

    total_t0 = time.time()
    results  = {}

    for name, script in SCRIPTS.items():
        if to_run[name]:
            success, elapsed = run_script(name, script)
            results[name] = (success, elapsed)

    # final summary
    print(f"\n{'='*60}")
    print("  FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Model':<16} {'Status':<10} {'Time':>8}")
    print(f"  {'-'*36}")
    for name, (success, elapsed) in results.items():
        status = "✓ DONE" if success else "✗ FAILED"
        print(f"  {name:<16} {status:<10} {elapsed:>6.1f}s")
    print(f"  {'-'*36}")
    print(f"  {'Total':<16} {'':<10} {time.time()-total_t0:>6.1f}s")
    print(f"{'='*60}")

    all_ok = all(s for s, _ in results.values())
    if all_ok:
        print("\n  All models trained successfully.")
        print("  Check each results folder for .xlsx and model files.")
    else:
        failed = [n for n, (s, _) in results.items() if not s]
        print(f"\n  Failed: {', '.join(failed)}")
        print("  Check the error output above for details.")
    print()


if __name__ == "__main__":
    main()

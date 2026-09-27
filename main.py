"""
main.py
-------
CLI entry point.

Usage:
    python sample_data.py                 # generates transactions.csv (once)
    python main.py "Analyze my last 3 months of spending and flag anything unusual"
    python main.py "Analyze spending" --data transactions.csv --out report.md
"""

import argparse
import os
import sys

from agent import FinSightAgent


def main():
    parser = argparse.ArgumentParser(description="FinSight — an agentic financial data analyst")
    parser.add_argument("goal", type=str, help="Natural language goal, e.g. 'Analyze my spending'")
    parser.add_argument("--data", type=str, default="transactions.csv", help="Path to transactions CSV")
    parser.add_argument("--out", type=str, default="report.md", help="Where to write the final report")
    parser.add_argument("--quiet", action="store_true", help="Suppress step-by-step trace output")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"Data file '{args.data}' not found. Run `python sample_data.py` first, "
              f"or pass --data pointing at your own CSV (columns: date, payee, category, amount).")
        sys.exit(1)

    agent = FinSightAgent()
    state = agent.run(goal=args.goal, data_path=args.data, verbose=not args.quiet)

    with open(args.out, "w") as f:
        f.write(state.final_report)

    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    print(state.final_report)
    print(f"\n(also written to {args.out})")


if __name__ == "__main__":
    main()

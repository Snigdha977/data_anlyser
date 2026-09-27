"""
sample_data.py
--------------
Generates a synthetic 3-month transactions.csv so the agent can be run and
demoed immediately without needing the user's real bank data. Includes a
few deliberate anomalies (a large one-off charge, a duplicate charge) so
detect_anomalies has something real to find.
"""

import random
from datetime import date, timedelta
import csv

random.seed(7)

PAYEES = [
    ("BigBasket Grocery", "groceries", (400, 1800)),
    ("Netflix Subscription", "subscriptions", (499, 499)),
    ("Spotify Premium", "subscriptions", (119, 119)),
    ("Uber Trip", "transport", (80, 450)),
    ("Indian Oil Petrol", "transport", (500, 2000)),
    ("Zomato Order", "dining", (150, 900)),
    ("Cafe Coffee Day", "dining", (120, 350)),
    ("Electricity Board", "utilities", (800, 2200)),
    ("Airtel Broadband", "utilities", (999, 999)),
    ("Amazon Shopping", "shopping", (300, 5000)),
    ("Flipkart Order", "shopping", (250, 4000)),
    ("Apollo Pharmacy", "healthcare", (100, 1200)),
    ("House Rent", "rent", (15000, 15000)),
]


def generate(out_path: str = "transactions.csv", start: date = date(2026, 5, 1), months: int = 3):
    rows = [("date", "payee", "category", "amount")]
    current = start
    end = date(start.year, start.month + months if start.month + months <= 12 else 1, 1) \
        if start.month + months <= 12 else date(start.year + 1, start.month + months - 12, 1)

    d = start
    while d < end:
        # income: salary on the 1st of each month
        if d.day == 1:
            rows.append((d.isoformat(), "Employer Salary", "income", 65000))

        # 0-3 random transactions per day
        for _ in range(random.choice([0, 0, 1, 1, 2, 3])):
            payee, cat, (lo, hi) = random.choice(PAYEES)
            amount = -round(random.uniform(lo, hi), 2)
            rows.append((d.isoformat(), payee, cat, amount))
        d += timedelta(days=1)

    # inject a couple of deliberate anomalies
    rows.append(("2026-06-14", "Unknown Electronics Store", "shopping", -18500.00))  # big outlier
    rows.append(("2026-07-02", "Airtel Broadband", "utilities", -999.00))            # duplicate charge
    rows.append(("2026-07-03", "Airtel Broadband", "utilities", -999.00))            # duplicate charge

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"Sample data written to {out_path} ({len(rows) - 1} transactions)")


if __name__ == "__main__":
    generate()

"""
tools.py
--------
Every real "action" the agent can take. Each tool is a plain Python function
with a docstring and a JSON-schema description (TOOL_SPECS) so the LLM can
decide when and how to call it. Keeping tools as pure functions (input in,
result out, no hidden state) makes the executor simple and makes each tool
independently testable.
"""

from __future__ import annotations
import json
import statistics
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def load_data(file_path: str) -> dict:
    """Load a CSV of transactions and return a schema summary."""
    df = pd.read_csv(file_path, parse_dates=["date"], dayfirst=False)
    df = df.sort_values("date").reset_index(drop=True)
    summary = {
        "n_rows": len(df),
        "columns": list(df.columns),
        "date_range": [str(df["date"].min().date()), str(df["date"].max().date())],
        "total_spend": round(float(df.loc[df["amount"] < 0, "amount"].sum()), 2),
        "total_income": round(float(df.loc[df["amount"] > 0, "amount"].sum()), 2),
    }
    # stash the dataframe itself in a module-level cache keyed by path so
    # later tool calls in the same run can reload it cheaply
    _DF_CACHE[file_path] = df
    return summary


_DF_CACHE: dict[str, pd.DataFrame] = {}


def _get_df(file_path: str) -> pd.DataFrame:
    if file_path not in _DF_CACHE:
        load_data(file_path)
    return _DF_CACHE[file_path]


# ---------------------------------------------------------------------------
# 2. Summary statistics
# ---------------------------------------------------------------------------

def compute_stats(file_path: str) -> dict:
    """Compute spend statistics: monthly average, std deviation, top categories."""
    df = _get_df(file_path)
    spend = df[df["amount"] < 0].copy()
    spend["month"] = spend["date"].dt.to_period("M")
    monthly = spend.groupby("month")["amount"].sum().abs()

    by_category = (
        spend.groupby("category")["amount"].sum().abs().sort_values(ascending=False)
        if "category" in spend.columns else pd.Series(dtype=float)
    )

    return {
        "monthly_spend": {str(k): round(float(v), 2) for k, v in monthly.items()},
        "avg_monthly_spend": round(float(monthly.mean()), 2) if len(monthly) else 0.0,
        "std_monthly_spend": round(float(monthly.std()), 2) if len(monthly) > 1 else 0.0,
        "top_categories": {str(k): round(float(v), 2) for k, v in by_category.head(5).items()},
        "single_txn_mean": round(float(spend["amount"].abs().mean()), 2) if len(spend) else 0.0,
        "single_txn_std": round(float(spend["amount"].abs().std()), 2) if len(spend) > 1 else 0.0,
    }


# ---------------------------------------------------------------------------
# 3. Anomaly detection
# ---------------------------------------------------------------------------

def detect_anomalies(file_path: str, z_threshold: float = 2.5) -> dict:
    """
    Flag transactions whose absolute amount is a statistical outlier
    (z-score based) and flag likely duplicate charges (same amount + payee
    within 3 days).
    """
    df = _get_df(file_path)
    spend = df[df["amount"] < 0].copy()
    spend["abs_amount"] = spend["amount"].abs()

    mean = spend["abs_amount"].mean()
    std = spend["abs_amount"].std() or 1e-9
    spend["z_score"] = (spend["abs_amount"] - mean) / std

    outliers = spend[spend["z_score"].abs() >= z_threshold]
    outlier_list = [
        {
            "date": str(row["date"].date()),
            "payee": row.get("payee", "unknown"),
            "amount": round(float(row["amount"]), 2),
            "z_score": round(float(row["z_score"]), 2),
        }
        for _, row in outliers.iterrows()
    ]

    # duplicate-charge heuristic: same payee + same amount within 3 days
    dupes = []
    if "payee" in spend.columns:
        grouped = spend.sort_values("date").groupby(["payee", "amount"])
        for (payee, amount), g in grouped:
            if len(g) < 2:
                continue
            dates = g["date"].tolist()
            for i in range(1, len(dates)):
                if (dates[i] - dates[i - 1]).days <= 3:
                    dupes.append({
                        "payee": payee,
                        "amount": round(float(amount), 2),
                        "dates": [str(dates[i - 1].date()), str(dates[i].date())],
                    })

    return {
        "n_flagged_outliers": len(outlier_list),
        "outliers": outlier_list,
        "possible_duplicate_charges": dupes,
    }


# ---------------------------------------------------------------------------
# 4. Categorization (rule-based fallback; LLM can override via categorize_llm)
# ---------------------------------------------------------------------------

_CATEGORY_KEYWORDS = {
    "groceries": ["grocery", "supermarket", "mart", "food"],
    "subscriptions": ["netflix", "spotify", "subscription", "prime"],
    "transport": ["uber", "ola", "fuel", "petrol", "metro", "transport"],
    "dining": ["restaurant", "cafe", "zomato", "swiggy", "dining"],
    "utilities": ["electricity", "water bill", "internet", "utility", "recharge"],
    "shopping": ["amazon", "flipkart", "mall", "shopping"],
    "rent": ["rent"],
    "healthcare": ["pharmacy", "hospital", "clinic", "medical"],
}


def categorize_transactions(file_path: str) -> dict:
    """Fill in missing 'category' values using simple keyword matching."""
    df = _get_df(file_path)
    if "category" not in df.columns:
        df["category"] = None

    filled = 0
    for idx, row in df.iterrows():
        if pd.notna(row.get("category")) and row.get("category") not in ("", "uncategorized"):
            continue
        payee = str(row.get("payee", "")).lower()
        assigned = "other"
        for cat, keywords in _CATEGORY_KEYWORDS.items():
            if any(kw in payee for kw in keywords):
                assigned = cat
                break
        df.at[idx, "category"] = assigned
        filled += 1

    _DF_CACHE[file_path] = df
    counts = df["category"].value_counts().to_dict()
    return {"n_categorized": filled, "category_breakdown": counts}


# ---------------------------------------------------------------------------
# 5. Charting
# ---------------------------------------------------------------------------

def plot_chart(file_path: str, out_path: str = "monthly_spend.png") -> dict:
    """Render a bar chart of monthly spend to out_path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = _get_df(file_path)
    spend = df[df["amount"] < 0].copy()
    spend["month"] = spend["date"].dt.to_period("M").astype(str)
    monthly = spend.groupby("month")["amount"].sum().abs()

    fig, ax = plt.subplots(figsize=(7, 4))
    monthly.plot(kind="bar", ax=ax, color="#4C72B0")
    ax.set_title("Monthly Spend")
    ax.set_ylabel("Amount")
    ax.set_xlabel("Month")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return {"chart_path": out_path, "months_plotted": list(monthly.index)}


# ---------------------------------------------------------------------------
# Tool registry + JSON schemas (consumed by llm_client for native tool-use)
# ---------------------------------------------------------------------------

TOOL_FUNCTIONS = {
    "load_data": load_data,
    "compute_stats": compute_stats,
    "detect_anomalies": detect_anomalies,
    "categorize_transactions": categorize_transactions,
    "plot_chart": plot_chart,
}

TOOL_SPECS = [
    {
        "name": "load_data",
        "description": "Load a transactions CSV and return row count, date range, and totals.",
        "input_schema": {
            "type": "object",
            "properties": {"file_path": {"type": "string"}},
            "required": ["file_path"],
        },
    },
    {
        "name": "compute_stats",
        "description": "Compute monthly spend statistics and top spending categories.",
        "input_schema": {
            "type": "object",
            "properties": {"file_path": {"type": "string"}},
            "required": ["file_path"],
        },
    },
    {
        "name": "detect_anomalies",
        "description": "Flag statistically unusual transactions and likely duplicate charges.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "z_threshold": {"type": "number", "description": "Z-score cutoff, default 2.5"},
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "categorize_transactions",
        "description": "Assign a spending category to each uncategorized transaction.",
        "input_schema": {
            "type": "object",
            "properties": {"file_path": {"type": "string"}},
            "required": ["file_path"],
        },
    },
    {
        "name": "plot_chart",
        "description": "Generate and save a bar chart image of monthly spend.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "out_path": {"type": "string"},
            },
            "required": ["file_path"],
        },
    },
]

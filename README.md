# FinSight — An Agentic Financial Data Analysis Agent

## 1. Problem / Task Chosen

**Data Assistant Agent (fintech-flavored):** given a transactions CSV and a
natural-language goal (e.g. *"Analyze my last 3 months of spending and flag
anything unusual"*), the agent autonomously plans a sequence of analysis
steps, executes real tools (pandas computations, anomaly detection,
categorization, charting), reflects on whether each step actually succeeded,
and returns a synthesized markdown report — not a single LLM prompt/response.

## 2. System Architecture

```
                     ┌─────────────────────┐
   user goal  ─────► │   1. PLANNER (LLM)   │  turns goal into an ordered
                     │                      │  list of tool names
                     └──────────┬───────────┘
                                │
                                ▼
                 ┌──────────────────────────────┐
                 │   for each planned step:      │
                 │                                │
                 │  2. ACT: LLM decides tool      │◄──┐
                 │     input args  → tool runs     │   │ retry (max 1x)
                 │                                │   │ if reflection
                 │  3. OBSERVE: LLM reflects on    │───┘ says "failed"
                 │     output (success? note)      │
                 └──────────────┬─────────────────┘
                                │  (AgentState accumulates
                                │   every step + result)
                                ▼
                     ┌─────────────────────┐
                     │  4. RESPONDER (LLM)  │  synthesizes full trace
                     │                      │  into final markdown report
                     └──────────┬───────────┘
                                │
                                ▼
                          final_report

Tools available to the agent:
  load_data · compute_stats · detect_anomalies ·
  categorize_transactions · plot_chart
```

**Design pattern:** this is a **Plan-and-Execute** agent with a lightweight
**ReAct-style reflection** step bolted on — the LLM doesn't just execute a
fixed pipeline; it plans which tools are relevant to the specific goal, and
after each tool call it judges whether the result actually helps and can
retry with corrected input if not. State (`AgentState` in `state.py`) is
threaded through the whole run so later steps and the final report can see
everything earlier steps produced.

### File layout

| File | Responsibility |
|---|---|
| `main.py` | CLI entry point |
| `agent.py` | The orchestrator — the Plan → Act → Observe → Respond loop |
| `llm_client.py` | All LLM calls (plan, decide tool input, reflect, write report) |
| `tools.py` | The actual tool functions + their JSON schemas |
| `state.py` | `AgentState` / `StepRecord` — shared memory across steps |
| `sample_data.py` | Generates a synthetic `transactions.csv` for demoing |

No agent framework (LangChain/CrewAI/etc.) is used — the orchestration loop
is plain Python (~80 lines in `agent.py`) so the core logic is transparent
and easy to defend as original work.

## 3. Setup / Run Instructions

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your API key
cp .env.example .env
# edit .env and paste your key, then:
export ANTHROPIC_API_KEY=sk-ant-...        # (or `set` on Windows)

# 3. Generate sample data (or point --data at your own CSV)
python sample_data.py

# 4. Run the agent
python main.py "Analyze my last 3 months of spending and flag anything unusual"
```

Your own CSV just needs these columns: `date, payee, category, amount`
(category may be blank — the agent will fill it in; `amount` is negative
for spend, positive for income).

## 4. Sample Input / Output

**Input:**
```bash
python main.py "Analyze my last 3 months of spending and flag anything unusual"
```

**Console trace (abridged):**
```
[PLAN] load_data -> compute_stats -> detect_anomalies -> categorize_transactions -> plot_chart

[STEP 1] load_data({'file_path': 'transactions.csv'})
          -> OK  (Data loaded successfully with a clear date range and totals.)

[STEP 2] compute_stats({'file_path': 'transactions.csv'})
          -> OK  (Monthly spend stats computed, giving average and std deviation.)

[STEP 3] detect_anomalies({'file_path': 'transactions.csv', 'z_threshold': 2.5})
          -> OK  (Found 1 outlier transaction and 1 possible duplicate charge.)

[STEP 4] categorize_transactions({'file_path': 'transactions.csv'})
          -> OK  (All transactions now have a category assigned.)

[STEP 5] plot_chart({'file_path': 'transactions.csv', 'out_path': 'monthly_spend.png'})
          -> OK  (Chart saved showing monthly spend trend.)
```

**Final report (`report.md`, abridged):**
```markdown
# Spending Analysis: May – July 2026

## Summary
- Average monthly spend: ~₹42,300 (σ ≈ ₹6,100)
- Total transactions analyzed: 210

## Anomalies Flagged
- **₹18,500 charge at "Unknown Electronics Store" (Jun 14)** — over 4 standard
  deviations above your typical transaction size. Worth verifying this was you.
- **Duplicate-looking charge**: "Airtel Broadband" ₹999 billed twice within
  2 days (Jul 2 and Jul 3) — check if you were double-billed.

## Top Spending Categories
1. Rent — ₹45,000
2. Shopping — ₹19,400
3. Groceries — ₹11,200
...

## Recommendation
Spending is stable month-to-month aside from the flagged outlier; if the
₹18,500 charge is legitimate, your effective average spend is within your
historical range.
```

A chart (`monthly_spend.png`) is also generated alongside the report.

## 5. Notes on Agentic Behaviour (for evaluators)

- **Planning**: the tool sequence is not hardcoded — it's generated per-goal
  by the LLM (`LLMClient.plan`), so a different goal (e.g. "just tell me my
  average monthly spend") produces a shorter plan.
- **Reasoning per step**: tool *arguments* are also decided by the LLM per
  step (`decide_tool_input`), using everything learned so far.
- **Tool use / execution**: real computation happens in `tools.py` via
  pandas — the LLM never fabricates numbers, it only decides which
  computation to run and interprets the results.
- **Observation / self-correction**: each step is reflected on
  (`LLMClient.reflect`); a failed step is retried once with corrected input
  before the agent moves on.
- **State across steps**: `AgentState` accumulates every step's output so
  later steps (and the final report) have full context, not just the
  immediately previous result.

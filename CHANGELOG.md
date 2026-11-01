# Changelog

## 0.2.0 (2026-11-01)

- **Ledger durable across a power cut.** WAL with `synchronous=FULL`, `fullfsync` and
  `checkpoint_fullfsync`. Under the old rollback journal a committed `sending` row could be
  rolled back after a power cut, and the request sent again. `tests/test_durability.py`
  simulates the cut with an `LD_PRELOAD` shim that keeps only what was fsynced.
- **Sender contract.** `Sender.send` must act at most once per `request_id`;
  `check_sender` tests a sender against a test destination.
- **Assisted provenance.** `decidedBy: "human_assisted"` marks a person who saw the agent's
  suggestion. Every label path refuses it; `labels --assisted`, `replay --assisted` and
  `check --assisted` keep and report it apart. It never chooses a threshold, counts toward a
  bar, or serves as a re-test.
- **Claude adapter** (`autonomy_gate.adapters.anthropic_messages:ClaudeMessagesAgent`)
  beside the OpenAI one: `output_config.format` JSON schema, explicit effort, refusal or
  `max_tokens` becomes a hold, enum lowercased, no fallback models.
- **Adapters off by default.** Neither builds a real client unless
  `AUTONOMY_GATE_ALLOW_MODEL_CALLS=1`. Both count tokens per call. The OpenAI adapter reads
  `AUTONOMY_GATE_OPENAI_EFFORT`.
- **`clock-test`.** Asks the same decisions with and without `decidedAt` and reports the
  paired difference, model, effort, date, n, tokens and cost. No real-model number is
  claimed.
- **README.** The history is corrected and dated; documented commands are parsed by a test.
- **CI.** The power-cut simulation is required, and a job runs both adapters through their
  real SDKs against a fake API on 127.0.0.1.

## 0.1.0 (2026-10-31)

- Replay against a logistic baseline at matched precision, a one-shot held-out re-test,
  shadow mode with go-live bars, and a guarded runtime with a kill switch and a write-ahead
  ledger.

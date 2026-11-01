# autonomy-gate

Decide when an LLM agent has earned the right to act on its own.

You give it decisions people already made and an agent (any Python callable). It:

1. **Replays** those decisions through the agent and a plain logistic-regression baseline,
   with the decision time in every request, and picks a confidence threshold from the
   labels.
2. **Re-tests** that frozen threshold once, on newer decisions it has never seen, and
   refuses to re-test on anything it was tuned on.
3. Runs the agent in **shadow** on live requests (it logs, it can't send) until the
   **go-live check** passes written bars, and names every bar that fails until then.
4. Lets it **act** through one guarded path: a kill switch read before every send, a cap,
   and a write-ahead ledger that settles a lost answer by looking it up, never by sending
   again.

Python 3.11+, no runtime dependencies, MIT. Everything in this repository runs on
synthetic data with a stand-in model, so it needs no API key.

## Why it exists

In October 2026 I had to decide whether the lead agent I built for my hauling business could
act without me. In order:

- **Oct 9, first release: a scorecard, not an LLM.** A logistic regression on 1,245 of my
  labeled decisions, split by time 60/20/20. On the newest fifth, held out, it approved 22
  posts and was right on 19 (0.86, Wilson 95% 0.67 to 0.95), at 10% recall. I had looked
  at that held-out set three times while fixing the policy, so 0.86 wasn't a clean
  estimate. The same release wrote the bars down: precision with a 95% lower bound and a
  minimum number of approvals, at least 7 days of shadow, and a shadow runner a test proves
  can't send.
- **Oct 9, later that day: GPT-5.5 decides.** Its first replay, on a stratified sample of
  200 decisions, also scored 0.86 precision, but caught only 75 of my 138 approvals (0.54).
  The two 0.86s come from different samples at different recall, so they aren't a paired
  comparison. Two causes: the request carried no clock, so the model judged "today's sale"
  against its own idea of today, and its rubric was narrower than I am. The same change
  added `decidedAt` and a wider rubric, and that version hasn't been replayed yet.
- **Oct 11: live approvals behind a kill switch, off by default.** Every decision the agent
  made carries `decidedBy`, so its approvals never become labels.

Two silent failures in those weeks are why a send here counts only with a receipt. They
were separate: a scheduled job printed its usage line and exited 0 every minute from Sep 17
until it was fixed on Oct 9, doing nothing; and from Sep 28 to Oct 2 the health pager's only
route was the one that was down, so nothing paged about the outage.

This package is that gate, rebuilt on synthetic data. None of my business's data is in it.

— Jacob Lopez

## What it checks that a single score won't

- **The clock.** Every record carries `decidedAt`, and every request the agent sees
  includes it. `autonomy-gate lint` fails a file whose records have no clock, or whose
  request snapshot was captured after the decision (`capturedAt > decidedAt`: facts from
  the future). At runtime, a rail holds any approval whose request has no clock.
- **Precision alone lies.** 9 right out of 10 is 0.90, but its Wilson 95% lower bound is
  0.60. The bars ask for both: precision ≥ 0.90 *and* a lower bound ≥ 0.85, at least 40
  overlaps with a person including 25 approvals, and a recall floor you choose.
- **The agent's own decisions poison the labels.** Every decision says who made it
  (`decidedBy: human | human_assisted | agent`). Every path that turns decisions into labels
  or agreement numbers refuses agent-made ones; `autonomy-gate labels` filters a raw export.
  Labels a person made from facts the agent couldn't know (`basis: operational`, such as a
  full schedule) are reported apart from content judgments.
- **A person who saw the suggestion isn't an independent label.** Once the agent's skips go
  to a person "with the suggestion attached", that person's call is `human_assisted`. It is
  kept in its own file (`labels --assisted`), reported apart in `replay --assisted` and
  `check --assisted`, never chooses the threshold, never counts toward a bar, and is never a
  re-test: every label path refuses it.
- **"Success" is a claim, not a fact.** A send counts as sent only with a receipt from the
  sender, or a receipt found by looking its request id up. An answer lost to a crash or a
  timeout ends `unconfirmed_never_resubmitted` and is never sent again.

## Two-minute demo

From a clone (it isn't on PyPI yet):

```sh
pip install -e ".[dev]"
```

The examples are pickup requests for a hauling business, made by
`autonomy-gate synth examples`. About a quarter of them are same-day ("Moving out today, need
a couch gone before 5pm"). The "person" who decided them is a formula, and the agent is
`autonomy_gate.synthetic:candidate`, a keyword-and-noise stand-in for a model. Its numbers
say nothing about any real model.

**1. Strip the clock, and the stand-in's recall collapses on the same-day requests.**

```text
$ autonomy-gate replay examples/leads_history.jsonl --candidate autonomy_gate.synthetic:candidate --bars examples/gate.yaml --strip-clock
500 human decisions, 334 approvals. CLOCK STRIPPED: no decidedAt.
as asked  the model's own answers, no threshold: approves 234, right 216: precision 0.923 [0.882, 0.951]; catches 216 of 334: recall 0.647 [0.594, 0.696]
candidate threshold 0.5982: ... misses: recall 0.647 < 0.70

By the agent's reason:
  reason                n  approve  hold  skip  person approved
  expired             138        0     0   138              108
```

Precision still looks fine. The stand-in, with no date to go on, calls 138 same-day
requests expired, and the person had approved 108 of them. That collapse is built into the
stand-in; whether a real model does the same is what `clock-test` (below) measures. Put the
clock back:

```text
$ autonomy-gate replay examples/leads_history.jsonl --candidate autonomy_gate.synthetic:candidate --bars examples/gate.yaml
500 human decisions, 334 approvals. Requests carried decidedAt.
candidate threshold 0.5982: approves 343, right 322: precision 0.939 [0.908, 0.960]; catches 322 of 334: recall 0.964 [0.938, 0.979]
          clears the bars
```

**2. Beat the baseline, lock the threshold, re-test once on newer decisions.**

```text
$ autonomy-gate replay examples/leads_history.jsonl --candidate autonomy_gate.synthetic:candidate --bars examples/gate.yaml --baseline --lock gate.lock
Baseline fitted on the oldest 250 decisions; replaying the newer 250.
Baseline held to the candidate's precision (0.937):
  threshold 0.8841: approves 48, right 45: precision 0.938 [0.832, 0.979]; catches 45 of 167: recall 0.269 [0.208, 0.341]
  approvals only the candidate caught: 119; only the baseline: 1; exact McNemar p = 1.82e-34
Locked threshold 0.5982 in gate.lock; the gate has now seen everything through 2026-02-18T14:46:00Z.

$ autonomy-gate retest examples/leads_history.jsonl --candidate autonomy_gate.synthetic:candidate --lock gate.lock
autonomy-gate: refused: refusing to re-test on data the gate has already seen; 500 ids were already used: ...

$ autonomy-gate retest examples/leads_heldout.jsonl --candidate autonomy_gate.synthetic:candidate --lock gate.lock
Held-out re-test at the frozen threshold 0.5982, 300 new decisions:
  approves 188, right 177: precision 0.941 [0.898, 0.967]
  catches 177 of 178: recall 0.994 [0.969, 0.999]
  PASSED. Recorded in gate.lock; this lock takes no other held-out set.
```

The re-test also refuses a request it has seen under a new id, and anything decided
before the newest decision the lock has seen. A second re-test on the same lock is refused:
change the prompt and you tune a new lock (`replay --previous-lock`) and re-test on newer
decisions.

**3. Shadow until the check goes GREEN.** One day of shadow:

```text
$ autonomy-gate shadow examples/leads_live.jsonl --candidate autonomy_gate.synthetic:candidate --lock gate.lock --log shadow.jsonl --until 2026-03-01T12:00:00Z
Shadowed 27 requests into shadow.jsonl (approve 19, skip 8). Nothing was sent.

$ autonomy-gate check --shadow shadow.jsonl --labels examples/leads_live_labels.jsonl --lock gate.lock
RED: 4 of 10 bars fail.
  FAIL  days of shadow                            0.9                 need >= 7
  FAIL  overlaps with a person                    15                  need >= 40
  FAIL  person approvals in overlaps              13                  need >= 25
  ok    precision                                 13/13 = 1.000       need >= 0.90
  FAIL  Wilson 95% lower bound                    0.772               need >= 0.85
  ...
```

Thirteen right out of thirteen is not enough evidence. After the rest of the period:

```text
$ autonomy-gate shadow examples/leads_live.jsonl --candidate autonomy_gate.synthetic:candidate --lock gate.lock --log shadow.jsonl
$ autonomy-gate check --shadow shadow.jsonl --labels examples/leads_live_labels.jsonl --lock gate.lock
GREEN: every bar passes.
  ok    days of shadow                            10.8                need >= 7
  ok    overlaps with a person                    174                 need >= 40
  ok    precision                                 101/105 = 0.962     need >= 0.90
  ok    Wilson 95% lower bound                    0.906               need >= 0.85
  ...
Skips: earned (60/62 = 0.968 on skip overlaps); they may close requests.
```

`check --lock` writes the result into the lock. The runtime acts only on a lock whose
re-test passed and whose latest check is GREEN, for the same candidate fingerprint.

**4. Act behind the kill switch, and crash mid-send.**

```text
$ R="examples/leads_live.jsonl --candidate autonomy_gate.synthetic:candidate --lock gate.lock --ledger ledger.db --switch switch.txt --live --outbox outbox.jsonl"
$ autonomy-gate run $R --batch 3
  lead-00799  held                            kill_switch: switch.txt is missing
  ...
$ autonomy-gate switch switch.txt on
$ autonomy-gate run $R --batch 8 --cap 2
  lead-00798  sent                            receipt
  lead-00802  sent                            receipt
  lead-00805  would_approve                   cap_reached
  lead-00803  skipped                         not_a_job
  ...
Ledger: held 3, sent 2, skipped 2, would_approve 4
$ autonomy-gate switch switch.txt off
$ autonomy-gate run $R --batch 3
  lead-00805  held                            kill_switch: switch.txt says 'stop'
  ...
Ledger: held 10, sent 2, skipped 2
$ autonomy-gate switch switch.txt on
$ autonomy-gate run $R --batch 5 --demo-crash before_write     # the process dies inside send()
$ autonomy-gate run $R --batch 1
  lead-00815  unconfirmed_never_resubmitted   found in sending at start
  lead-00818  sent                            receipt
```

Deferred approvals (`would_approve`) go out on a later cycle from the stored verdict,
without asking the model again, while the switch is on. A cycle that finds the switch off
hands them to a person instead: above, the four waiting approvals end `held` with the three
new requests. `--demo-crash after_write` shows the other case: the send happened, so the
lookup finds its receipt and the item ends `sent`, still exactly once.

**5. Put one agent decision into a label file.** Change any line of
`examples/leads_history.jsonl` to `"decidedBy": "agent"` and run `pytest`:
`tests/test_provenance.py` fails, and so do `autonomy-gate lint` and every command that
reads labels.

## Data

One JSON object per line (wrapped here to read):

```json
{"id": "lead-00012",
 "decidedAt": "2026-03-04T15:20:00Z",
 "capturedAt": "2026-03-04T15:20:00Z",
 "request": {"text": "Moving out today, need a couch gone before 5pm.",
             "postedAt": "2026-03-04T13:05:00Z",
             "features": {"distance_miles": 4.2, "has_photo": true, "items": 3, "replies": 1}},
 "decision": {"action": "approve", "decidedBy": "human", "basis": "content"}}
```

`request` is yours: whatever the agent should see. The baseline reads the numeric and
boolean `features`. Live requests for `shadow` and `run` have no `decision`.

`decidedBy` is `human` for a person who decided without seeing the agent's answer,
`human_assisted` for a person who saw it first (anything decided from an `advisory_skip`,
or from a hold that showed the agent's verdict), and `agent` for the agent itself. Split a
raw export with `autonomy-gate labels export.jsonl -o labels.jsonl --assisted
assisted.jsonl`.

## Your agent

Any callable from a request dict to a verdict:

```python
def my_agent(request: dict) -> dict:
    # request: {"id", "decidedAt", ...everything in your record's "request"}
    return {"action": "approve", "reason": "same_day_job", "confidence": 0.93}

my_agent.fingerprint = "my-agent prompt v3"  # optional; changes invalidate the lock
```

`approve` acts, `hold` leaves it for a person, `skip` closes it. Pass it as
`--candidate mypackage.agent:my_agent` (or `path/to/file.py:my_agent`). An exception, a
timeout (`--timeout`) or anything that isn't a valid verdict becomes a hold.

### Model adapters (optional, off by default)

No adapter calls a model API unless you set `AUTONOMY_GATE_ALLOW_MODEL_CALLS=1`. Without
it, building one fails before any SDK is imported, so a stray key in your environment can't
spend money from a test or a demo. Nothing in the package root imports an adapter.

### OpenAI Responses API

With `OPENAI_API_KEY`, `AUTONOMY_GATE_OPENAI_MODEL` (and, if you want one,
`AUTONOMY_GATE_OPENAI_EFFORT`) set in your environment:

```sh
pip install -e ".[openai]"
AUTONOMY_GATE_ALLOW_MODEL_CALLS=1 autonomy-gate replay labels.jsonl --candidate autonomy_gate.adapters.openai_responses:OpenAIResponsesAgent
```

It asks for a strict JSON-schema verdict with Structured Outputs (`text.format`,
`strict: true`) and `store: false`, and treats a refusal or an `incomplete` response as an
error, so the gate holds. Your own instructions can go in the file named by
`AUTONOMY_GATE_INSTRUCTIONS_FILE`. It follows OpenAI's
[Structured Outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs)
and the [Responses API reference](https://developers.openai.com/api/reference/resources/responses/methods/create),
both read on 2026-10-31.

### Claude (Anthropic Messages API)

With `ANTHROPIC_API_KEY` set (the model defaults to `claude-opus-5-5`; override it with
`AUTONOMY_GATE_ANTHROPIC_MODEL`, and the effort, which defaults to `low`, with
`AUTONOMY_GATE_ANTHROPIC_EFFORT`):

```sh
pip install -e ".[anthropic]"
AUTONOMY_GATE_ALLOW_MODEL_CALLS=1 autonomy-gate replay labels.jsonl --candidate autonomy_gate.adapters.anthropic_messages:ClaudeMessagesAgent
```

It puts the verdict schema in `output_config.format` (`type: "json_schema"`) and always
sends the effort. A `refusal` or `max_tokens` stop reason, or any stop other than
`end_turn`, raises, so the gate holds. `action` is lowercased before the verdict is
checked, because structured outputs don't promise an enum's capitalization, and the schema
carries no numeric bounds (unsupported there), so `Verdict` checks that confidence is in
[0, 1]. It sends no fallback models: an answer from a model the lock never tested must not
act. It follows Anthropic's
[structured outputs guide](https://platform.claude.com/docs/en/build-with-claude/structured-outputs).

Both adapters add every response's token usage, refusals included, to `agent.usage`. The
tests drive them with fake clients, and `tests/test_adapters_loopback.py` runs each through
its real SDK against a fake API on 127.0.0.1 with a fake key (a CI job installs both SDKs
for it). This repository has not called either live API.

## The runtime in your code

```python
from autonomy_gate import FileKillSwitch, GateLock, Guard, Ledger
from autonomy_gate.cli import agent_fingerprint

class MySender:
    def send(self, request_id, payload):   # use request_id as the idempotency key
        ...                                # return a receipt id, or raise
    def lookup(self, request_id):          # the receipt if it happened, else None
        ...

fp = agent_fingerprint(my_agent, "mypackage.agent:my_agent", True)
guard = Guard(my_agent, Ledger("ledger.db"), FileKillSwitch("switch.txt"),
              GateLock.load("gate.lock"), fingerprint=fp, sender=MySender(),
              max_sends_per_cycle=20)
guard.cycle(new_records)   # call once per cycle
```

`fp` has to match the fingerprint `replay --lock` stored for the same `--candidate`, or
`Guard` refuses to go live; `agent_fingerprint` computes it the same way the CLI does.

`send` must act at most once per `request_id`: a second call with an id that already acted
returns the first receipt and does nothing else. The ledger is what stops a second send; the
idempotency key is the backstop if a ledger row is ever lost. Check your sender against a
test destination before going live:

```python
from autonomy_gate import check_sender

check_sender(MySender(), actions=count_messages_in_test_inbox)  # raises SenderContractError
```

Leave `sender` out and nothing can be sent: approvals are held (`live_off`). Rails are
functions `(request, verdict) -> verdict | None`; the most cautious verdict wins and a rail
can never turn a hold or a skip into an approval. `autonomy_gate.isolation.assert_isolated`
walks a module's imports and fails if it can reach network or send code; this repository
uses it to prove that the policy and shadow modules' own imports can't reach the actuator,
and you can use it on your own policy module.

| Ledger state | Meaning |
| --- | --- |
| `held` | A person decides (hold, model error, below threshold, a rail, live off, switch off). |
| `advisory_skip` | The agent would close it, but skips haven't earned that yet. |
| `skipped` | Closed without acting, once skips cleared their own bar. |
| `would_approve` | Approved but over the cap; sent later without a second model call. |
| `sending` | Intent committed before the sender is called. |
| `sent` | A receipt from the sender or from a lookup. |
| `unconfirmed_never_resubmitted` | The answer was lost and the lookup couldn't confirm it. Never resent. |

## The bars

`examples/gate.yaml` (`name: number` lines and `#` comments; anything else is an error):

| Bar | Default | |
| --- | --- | --- |
| `min_shadow_days` | 7 | span of `decidedAt` in the shadow log |
| `min_overlaps` | 40 | shadowed requests a person also decided |
| `min_overlap_approvals` | 25 | of those, ones the person approved |
| `min_precision` | 0.90 | of the agent's approvals, share the person approved |
| `min_precision_lower_bound` | 0.85 | Wilson lower bound of that share, at `confidence` |
| `min_recall` | 0.0 (0.70 in the example) | of the person's approvals, share the agent caught |
| `max_p95_latency_s` | 60 | 95th percentile of the time each model call took |
| `max_unavailable` | 0.05 | share of model calls that failed or timed out |
| `skip_min_precision`, `skip_min_overlaps` | 0.80, 50 | before a skip may close a request |
| `confidence` | 0.95 | for every interval |

The defaults are the bars the author wrote for his own agent, except `min_recall`, which is
yours to choose.

## Measuring the clock on a real model

`autonomy-gate clock-test` asks the same decisions twice, once with `decidedAt` in every
request and once without it, and nothing else different. It compares the model's own
answers (no threshold, so nothing tuned on one arm favors it), pairs the arms on the
person's approvals with an exact McNemar test, and splits recall by whether the request
uses relative time words. The report records the model and effort, the date, how many
decisions, a hash of the label file, the candidate's fingerprint, and the tokens each arm
used, priced if you pass prices.

```sh
pip install -e ".[openai]"
export OPENAI_API_KEY=... AUTONOMY_GATE_OPENAI_MODEL=gpt-5.5 AUTONOMY_GATE_OPENAI_EFFORT=low
AUTONOMY_GATE_ALLOW_MODEL_CALLS=1 autonomy-gate clock-test examples/leads_history.jsonl \
  --candidate autonomy_gate.adapters.openai_responses:OpenAIResponsesAgent \
  --sample 200 --workers 4 --timeout 60 \
  --usd-per-mtok-in <input price> --usd-per-mtok-out <output price> \
  --out reports/clock-gpt-5.5-low.json
```

The same works with `--candidate autonomy_gate.adapters.anthropic_messages:ClaudeMessagesAgent`
and `ANTHROPIC_API_KEY`. A sample of 200 is 400 calls. The synthetic "person" is a formula,
so precision and recall here measure agreement with that formula; the difference between
the two arms is the part that speaks to the clock.

Both arms get the same instructions. The adapters' default instructions say what to do *if*
the request includes `decidedAt`; instructions of your own that say every request has it
could push the bare arm toward holds. Only `decidedAt` is removed: the synthetic requests
keep `postedAt`, usually minutes to a few hours before the decision, so the bare arm still
knows when the post went up, just not when it was decided.

This repository claims no real-model number. The only figures above come from the
stand-in. Run it with your key and commit the report; if a real model doesn't lose recall
without the clock, the report says so.

## Measuring your own agent

The synthetic numbers above measure nothing real. To get real ones:

- **Replay**: export decisions people made, with the time each was made and the request as
  it looked then (`capturedAt` no later than `decidedAt`). Run `lint`, then `replay
  --baseline`. Freeze the prompt (its fingerprint goes in the lock) before you label the
  held-out set.
- **Held-out**: label newer decisions without looking at the model's calls, then `retest`
  once. Report whole counts with the intervals, as the tool prints them.
- **Latency and availability**: the shadow log times each model call from the caller's side
  (`latency_s`) and records every failure (`error`). That is what the p95 and unavailable
  bars read, so run shadow on the same model, region and timeout you'll use live.
- **Live**: count outcomes from the ledger, not from your job's own success messages.

## What was tested

- `pytest`: 154 tests in `tests/`, all passing locally on Python 3.11, 3.12 and 3.13 with
  both SDKs installed and the power-cut simulation required (151 pass and 3 skip where the
  SDKs aren't installed). The CI workflow runs them on the same three, plus a job with both
  SDKs. They cover the Wilson interval against known values, threshold choice, the clock
  and leakage lint, provenance on every label path (including the committed example files
  and assisted decisions), the clock-stripped replay, the baseline comparison, held-out
  reuse refusal by id, by content and by time, shadow isolation (no sender parameter, no
  socket opened, imports walked), the go-live bars, kill switch, cap, skips, a sender that
  raises or returns no receipt, the sender contract, a real process killed with `os._exit`
  inside `send()` before and after the write, a simulated power cut after the `sending`
  commit, both adapters against fake clients and through their real SDKs against a fake API
  on 127.0.0.1, `clock-test` end to end, every command the README shows, and the CLI demo.
- `ruff check` and `ruff format --check`: clean.
- Scanned every tracked file, and every file in every commit, with `detect-secrets` 1.5.0:
  no findings.

Not tested: a real model (no live API was called), a real sender over the network, a power
cut on real hardware or on macOS (the simulation needs Linux), or several runners sharing
one ledger (run one runner per ledger).

## Limits

- The stand-in model shares its formula with the synthetic person, so its scores are
  optimistic by construction. Even so, the bars are tight: of seeds 1 to 8, five end the
  shadow check GREEN and three stay RED on the Wilson lower bound (0.831 to 0.837, against
  0.85). The examples use seed 3.
- The isolation check reads source. It follows imports inside your package, including ones
  inside functions, and judges everything else by name. It doesn't see `importlib` or
  `__import__` calls, and it doesn't follow a parent package's `__init__`: this package's
  `__init__` imports the actuator, so `import autonomy_gate.shadow` loads it, though
  nothing in shadow refers to it. It finds modules with `importlib.util.find_spec`, which
  imports their parent packages, so point it only at code you would import anyway.
- A timed-out model call keeps running in a background thread; its answer is ignored.
- `ledger.db` is SQLite in WAL mode with `synchronous=FULL` (plus `fullfsync` and
  `checkpoint_fullfsync`, which macOS needs to really flush), so a committed `sending` row
  survives a power cut as well as a crash; `tests/test_durability.py` simulates the cut by
  keeping only what was fsynced. WAL needs a local disk, and the ledger takes one runner. A
  send and its ledger row still can't be one transaction, which is why a lost answer is
  settled by lookup, and why `send` must be idempotent on `request_id`.

## License

MIT. See [LICENSE](LICENSE).

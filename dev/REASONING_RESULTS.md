# Reasoning phase results and outstanding experiments

Status as of 2026-09-20: implementation and offline tests verified; SFT smoke
passed. GPU RL smoke and the full ablation campaign have not run. The user's
other LLM training takes priority. No correctness improvement is claimed.

## Measurements available

| Measurement | Result | Scope |
|---|---|---|
| Original d24 SFT checkpoint | step 486 | Existing parent, unmodified |
| Original bounded ChatCORE | .2056 |24 examples per task,1024-token budget |
| Original bounded GSM8K |0/24 | Small baseline sample |
| Reasoning SFT smoke |50 iterations | Fresh optimizer, .3 LR fraction |
| Raw training loss |.552706 → .435255 | First/last five-step means |
| Validation BPB |.344775 | New reasoning mixture; no historical-BPB comparison |
| Training time |69.918s | Excludes preparation/load/save |
| Warmed throughput |12,396 packed tokens/s | Median, batch16384 tokens |
| Peak SFT allocated VRAM |22.16GiB | One RTX PRO 6000 Blackwell |
| Closed think blocks after reload |18/18 | Two completed prompts before interruption |
| Binary correctness in partial reload check |0/18 completions | Not a completed test evaluation |
| Final CPU-only suite |190 passed,14 skipped | CUDA hidden to protect other training |

The smoke mixture had 11,255 training rows and 192 validation rows, including 25%
unchanged replay. This run used small MetaMath/OpenMath/procedural caps, not the
full cold-start dataset. The checkpoint is `reason_sft/d24-reason-smoke-20260919`
step 50. Commands, input configuration, dataset provenance and limitations are
recorded in [REASONING_LOG.md](REASONING_LOG.md).

## What remains before a budget decision

Run the five-step GPU RL smoke (four prompts per step, eight samples each) when
the priority training is finished. Verify real-GPU gradients, component logs,
zero-variance handling and checkpoint reload, then measure RL and evaluation
costs. The SFT rate corresponds to roughly 80.7 seconds per million packed tokens
after warmup, but does not estimate the full campaign's inference-dominated cost.
Do not present the interrupted, contended evaluation as isolated-GPU throughput.

Every future project GPU run must be monitored with an 80% total-device VRAM
ceiling. Cancel only the project's owned process group if it crosses that limit;
never alter the user's priority training. The single-GPU runner includes this
guard. Full runs still require the user's measured campaign budget decision;
each run projected above approximately six GPU-hours needs separate approval.

## Planned comparison, not measured results

| Arm | Treatment | Status |
|---|---|---|
| A | Original SFT plus unchanged `chat_rl`, one seed | Not run |
| B | Full reasoning SFT | Smoke only |
| C | Shared B plus binary reasoning RL, seeds 42/43 | Not run |
| D | Shared B plus shaped reasoning RL, seeds 42/43 | Not run |

Match C/D prompt seeds, sampling, curriculum rules and rollout budgets. Record
attempted rollout steps separately from optimizer updates. Document A's existing
256-token training cap and other baseline differences. Evaluate full GSM8K
greedy pass@1 and sampled pass@8, fixed procedural held-outs at difficulties 1/3,
and all five ChatCORE tasks, with identical evaluation budgets across arms.

The requested 20 deterministic random D-arm completions cannot be inspected until
the D runs exist. Keep that exploitation audit and the uncertainty/computation
report as explicit completion gates. Neither the short smoke nor exact question
deduplication establishes correctness gains or absence of semantic contamination.

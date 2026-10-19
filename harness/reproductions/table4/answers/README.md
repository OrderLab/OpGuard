# Maintainer answers and historical controls

Do not provide this directory or its Git history to a diagnostic evaluator. It
contains known causes, supplied repairs, issue discussions and expected outcomes.
Use `prepare_inputs.py` here to export the separate `../inputs/` templates; the
export does not include this directory. Existing study captures can contain these
historical scripts and are not suitable as unhinted evaluation inputs.

## Reduced Table 4 workloads

These scripts retain the workloads and controls used to investigate Bugs 21–28.
They run historical upstream implementations with smaller inputs/models. They
are not a complete replay of all eight original paper experiments. In particular,
24 required a different routing configuration on the test machine, and 26's
paper-to-issue mapping remains unconfirmed.

| Case | Workload / upstream source | Scope of observed reproduction |
| --- | --- | --- |
| 21 | `workload_zero.py`, DeepSpeed 0.9.5; [Neox #1004](https://github.com/EleutherAI/gpt-neox/issues/1004) | Two-GPU ZeRO-2 with weighted embedding gradients and reduction-stream delay; adding the missing wait restores the reference result. |
| 22 | `workload_context.py`, native DeepSpeed 0.9.1 CUDA extension; [#3404](https://github.com/deepspeedai/DeepSpeed/issues/3404) | Repeated single-token inputs after a seven-token prompt; resetting the hidden cache restores repeatability. |
| 23 | `prepare_cases.py`, pinned Megatron vocabulary-padding helper; [#1754](https://github.com/NVIDIA/Megatron-LM/issues/1754) | Logical TP 1 versus 2 gives vocabulary sizes 50257 versus 50258 and different losses on otherwise zero logits. |
| 24 | `workload_dispatcher.py`, Megatron commit `81fee9b0047fb3ac6001b5e71e4df89fc01b2a1c`; [#1327](https://github.com/NVIDIA/Megatron-LM/issues/1327) | Actual all-gather/all-to-all dispatchers, four GPUs, an empty expert, top-k=3: same forward and output gradients, different input gradients. The reduced top-k=2 test did not diverge on GB200. |
| 25 | `workload_zero.py`, DeepSpeed 0.14.2; [#5545](https://github.com/deepspeedai/DeepSpeed/issues/5545) | Two-GPU ZeRO-1 overlap race, original workload detected by CUDA Stream Sanitizer; weighted/delayed control also changes final loss and weights. |
| 26 | `workload_long_context.py`, native DeepSpeed 0.7.0; candidate [#2243](https://github.com/deepspeedai/DeepSpeed/issues/2243) | Length 192 exceeds cache stride 128; the capacity-256 control removes nondeterminism. This reproduces the candidate issue, not a confirmed paper mapping. |
| 27 | `prepare_cases.py`, SmolVLMVisionEmbeddings 4.54.1 versus 4.55.1; [#41190](https://github.com/huggingface/transformers/issues/41190) | Shared random weights and a 27×27 patch grid expose position-grid rounding differences. |
| 28 | `workload_qwen.py` / `workload_qwen_state.py`, Transformers commit `62db3e6ed67a74cc1ed1436acd9973915c0a4475`; [#36009](https://github.com/huggingface/transformers/issues/36009) | Whole pinned Qwen2-VL implementation with a tiny random model; a generate prelude leaves `rope_deltas` that changes direct-forward positions. Clearing only that state restores the generate reference. |

`evidence/` contains the retrieved issue metadata and discussion. The workspace
alias probe in `workload_workspace.py` is an additional exploratory control around
DeepSpeed PR #3256; it is not the evidence used to claim case 26's reproduction.

## Environment and inputs

Use a separate virtual environment with editable OpGuard and OpGuard-harness
installs. The recorded environment used Python 3.11.2, PyTorch 2.9.1+cu130,
Transformers 4.49.0, NumPy 1.26.4, CUDA toolkit 13.1 and four GB200 GPUs on aarch64.
Historical CUDA sources are built separately in a task-local extension directory.
`build_inference.py` defaults to SM 10.0; set `TORCH_CUDA_ARCH_LIST` for other GPUs.
These scripts do not download model weights or install historical libraries into
the base environment.

Keep upstream checkouts, builds and generated captures outside this repository.
Select historical DeepSpeed/Megatron checkouts with `PYTHONPATH` when running
their workloads. The ZeRO workload contains import compatibility shims for the
newer PyTorch elastic API. Distributed workloads use `torchrun --nproc-per-node=2`
(21/25) or `4` (24); CUDA stream races can depend on scheduling.

Preserve runtime settings from the failing workload. In the follow-up 21/25
study, the experiment preparation script accidentally copied
`CUDA_DEVICE_MAX_CONNECTIONS=1` from an unrelated configuration. It masked both
races even without OpGuard. With the setting fixed at `8`, both workload-only and
full DeepSpeed-source captures retained the divergent loss and weights. This was
an experiment setup error; neither tool sets that variable automatically.

| Connection count | No OpGuard capture | Workload-only capture | DeepSpeed-source capture |
| --- | --- | --- | --- |
| 1 | Reference loss/weights match bytewise | Match | Match |
| 8 | Loss/weights diverge | Diverge | Diverge |

These results apply to both reduced weighted/delayed workloads above. With
connections fixed at `8`, adding the missing compute-stream wait for the previous
reduction restores bytewise reference loss and weights in both the untraced and
full-source capture controls. The latter have no aligned diff; unresolved calls
remain unresolved, so the saved tensor outputs provide the independent check.
CUDA Stream Sanitizer identifies the buffer overwrite and pending reader; the
trace's first observed differences are downstream gradient-norm observations,
not direct proof of the race. CUDA's
[connection documentation](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/05-appendices/environment-variables.html#cuda-device-max-connections)
explains how streams sharing a work queue can acquire false dependencies. Setting
the connection count to `1` is a diagnostic perturbation, not a synchronization fix.

`prepare_cases.py ROOT` expects `ROOT/vendor/raw/tokenizer.py` from the issue-linked
Megatron revision and `smolvlm-4.54.1.py` / `smolvlm-4.55.1.py` from the respective
Transformers tags. It extracts the original definitions without changing their
line numbers and creates the case 23/27 workloads and harness specs.
Qwen workloads expect the pinned `modeling_qwen2_vl.py` alongside the script.
The remaining workload setup and baseline/candidate specs are explicit experiment
inputs, not automatically generated by this directory.

## Running and checking evidence

For example, after preparing a DeepSpeed 0.9.1 source tree:

```bash
python reproductions/table4/build_inference.py "$CASE_ROOT/vendor/deepspeed-0.9.1" \
  --build "$CASE_ROOT/extensions/ds091"
python reproductions/table4/answers/workload_context.py \
  --extension "$CASE_ROOT/extensions/ds091" --reset --output "$CASE_ROOT/case22/raw-baseline.pt"
python reproductions/table4/answers/workload_context.py \
  --extension "$CASE_ROOT/extensions/ds091" --output "$CASE_ROOT/case22/raw-candidate.pt"
```

Run a prepared baseline/candidate spec through the public harness API:

```bash
python reproductions/table4/run_experiment.py "$CASE_ROOT/case22/spec.json" \
  --root "$CASE_ROOT/experiments"
```

The runner saves every attempt and supports `--resume EXPERIMENT_ID`. Successful
report generation is not a numerical verdict. Inspect `first_diff`, the pairing
basis, unresolved regions and source context using the repository's skills.

`verify_reproductions.py ROOT` checks an existing eight-case result tree, including
the named raw stress/control outputs in the script. It reports reduced-scope
results and explicitly leaves `all_original_paper_cases_fully_reproduced` false.
`verify_skill_validation.py VALIDATION_DIR` verifies SHA-256-bound tensor snapshots
from an `evidence-manifest.json`, then checks the cache-reset controls for 22/28
with bytewise comparisons. Snapshot files and the manifest are experiment outputs
and must already exist. These are workload observations, not a harness tensor API.

The 22/28 skill assessment was a walkthrough with known issue hypotheses, followed
by source inspection and controlled tests; it was not a blind diagnostic benchmark.
Re-exporting saved captures with the conservative alignment recovered the native
`softmax_context_fp32` boundary for 22. For 28's different API paths it retained
undetermined regions; the same-path state control paired successfully. The agent
still needs caller/callee and upstream/downstream source reasoning to establish
the root cause.

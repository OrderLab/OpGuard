# Reproduction inputs and maintainer answers

`inputs/` holds diagnostic workload templates without supplied repairs, causal
explanations or issue identifiers. `answers/` holds the historical controls,
verifiers, issue evidence and interpretation. Its scripts are for maintainers,
not solver input. The generic capture runner and native extension builder remain
in this directory.

Prepare a separate input bundle and private provenance directory:

```bash
python reproductions/table4/answers/prepare_inputs.py STUDY_ROOT \
  --output /path/to/diagnostic-inputs --private-output /path/to/private-records
```

The preparer exports an explicit file allowlist through
`opguard_harness.input_bundle.export_inputs`. The public inventory contains only
relative paths and hashes. Case mappings and origin paths are written separately.
The private output also receives a snapshot of the maintainer answer directory.
No Git history, previous captures, diagnostics, repair controls or answer scripts
are copied. The two tasks whose previous executable references were patched
implementations receive saved output tensors instead; their patched code and
extensions remain private. Six other tasks retain ordinary executable reference
configurations. This preserves the distinction between an output reference and a
traceable reference implementation.

Keep the resulting bundle separate from this repository and the old study tree.
Old captures contain embedded source snapshots; moving answer files or editing
templates does not sanitize those captures. Regenerate captures from the exported
inputs. Do not alter historical artifacts to conceal their provenance.

This export is an input boundary, not an OS sandbox or proof that every selected
upstream comment is free of prior knowledge. Review the allowlist and sources,
use a new conversation and expose only the bundle plus normal runtime packages.
Do not mount the maintainer repository or private records into that workspace.

Maintainers can validate a copy without adding results to the delivered inputs:

```bash
python reproductions/table4/answers/check_inputs.py /path/to/private-records \
  --workdir /path/to/validation-copy --capture 21 25
```

This checks reproduction preservation and the new captures' embedded source; it
does not claim a blind diagnosis. Its logs and known-outcome checks stay private.

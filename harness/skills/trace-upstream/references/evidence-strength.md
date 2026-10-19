# What the available evidence can establish

Distinguish these observations before deciding whether an input already differs:

| Evidence | What agreement supports | What it cannot establish |
| --- | --- | --- |
| Shape/dtype/layout | Compatible representation | Equal values or correct alignment |
| Mean/norm/sample | Agreement of that statistic/sample | Equality of the whole tensor |
| XOR or another compact fingerprint | Agreement of the recorded fingerprint | Collision-free equality or value provenance |
| Full tensor snapshot | Equality under an explicit comparison at capture time | What a concurrent consumer later read |
| Versioned producer/mutation record | A specific observed data lineage | Unrecorded writes or complete synchronization |
| Embedded source | What that snapshot expresses | Which branch/config/patch actually executed |

For example, bit patterns `[1, 1]` and `[2, 2]` have the same XOR, zero, despite
being different arrays. Do not interpret zero XOR as an all-zero tensor, or
subtract two fingerprint numbers to compute a numerical error. A differing
fingerprint supports different captured representations only under a compatible,
correctly executed fingerprint procedure.

Current diagnostic captures default to four XORs over contiguous regions of the
logical flattened byte stream. Read `xor_sig_layout` for the scheme and exact
half-open byte ranges; legacy scalar signatures and row-based vectors do not
have interchangeable partitions. `regional_comparison.differing_regions`
identifies regions with observed fingerprint differences, not differing element
counts. Agreement in the other regions remains a lossy observation.

A tail-region difference can suggest padding or unwritten storage. Establish the
valid extent and actual consumer reads from source and controlled evidence;
padding can also be read incorrectly and cause the bug. Do not discard the last
region or zero-fill allocations to make captures agree. Keep such experiments
separate from the original capture and retain any rejected causal hypothesis.

Use three input states: **difference observed**, **agreement under a stated
criterion**, and **unknown**. Equal lossy summaries belong to the third state for
whole-tensor equality. They do not justify moving the entire investigation into
"same inputs, different operator outputs".

Walk the consumer's source expressions backward when dynamic producer IDs are
missing, but label those edges `source_inference`. Views, casts, `.data`, mutation,
RNG, state reuse, communication and ordering can all matter; follow what the code
and observations support, without privileging any one bug family.

A previous exporter-OK call is merely a neighboring observation. It may consume
another tensor, inspect only overflow, omit inputs or have a colliding fingerprint.
It is not automatically a clean upstream boundary for the differing value.

Keep source definitions separate from execution evidence. A bundled repair
function is a proposed intervention until captured arguments/configuration or
runtime observations establish that it ran. An unversioned shared source snapshot
cannot establish both runs' exact code/configuration.

When evidence ends, name the missing edge: for example a value before conversion,
a versioned alias relationship, actual launch configuration, a paired invocation,
or ordering between a write and read. Request that observation rather than naming
a root cause from proximity alone.

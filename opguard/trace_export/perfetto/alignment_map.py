"""Apply artifact-bound correspondence reviews to a new Perfetto view.

All events are addressable, including events from different trace files. The map
contains agent assertions with evidence references, not independently certified
semantic equivalence. No workload execution, tensor conversion or rematching.
"""
import argparse
import collections
import copy
import gzip
import hashlib
import io
import json
from pathlib import Path

MAX_BYTES = 256 * 1024 * 1024
DECISIONS = {"comparable", "not_comparable", "undetermined"}
BASES = {"observed", "source_inference", "intervention_tested", "externally_reported"}


def read_export(path):
    path = Path(path).expanduser().resolve()
    with path.open("rb") as stream:
        encoded = stream.read(MAX_BYTES + 1)
    if len(encoded) > MAX_BYTES:
        raise ValueError("Trace exceeds the encoded size limit")
    stream = gzip.GzipFile(fileobj=io.BytesIO(encoded)) if path.suffix == ".gz" else io.BytesIO(encoded)
    with stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("Trace exceeds the decoded size limit")
    doc = json.loads(raw)
    if not isinstance(doc, dict) or not isinstance(doc.get("traceEvents"), list):
        raise ValueError("Expected a Perfetto traceEvents export")
    return doc, hashlib.sha256(encoded).hexdigest()


def tensor_slot(args, path):
    if (not isinstance(path, list) or len(path) != 2 or path[0] not in ("inputs", "outputs")
            or type(path[1]) is not int or path[1] < 0):
        raise ValueError("Tensor slot must be [inputs|outputs, nonnegative index] in exported io")
    values = (args.get("io") or {}).get(path[0], [])
    if path[1] >= len(values) or not isinstance(values[path[1]], dict):
        raise ValueError("Tensor slot is absent from exported io")
    return values[path[1]]


def validate_reviews(doc, sha, mapping):
    if (not isinstance(mapping, dict) or mapping.get("version") != 1
            or mapping.get("artifact_sha256") != sha):
        raise ValueError("Alignment map version or artifact SHA-256 does not match")
    events = doc["traceEvents"]
    if any(e.get("ph") == "B" and "status" in e.get("args", {}) for e in events):
        raise ValueError("Alignment maps require flat operation exports")
    ops = {i: e for i, e in enumerate(events) if e.get("ph") == "X"
           and e.get("cat") == "opguard" and "status" in e.get("args", {})}
    ids = collections.Counter(e["args"].get("record_id") for e in ops.values())
    if None in ids or any(count != 1 for count in ids.values()):
        raise ValueError("Alignment maps require unique operation record IDs")

    def index(ref):
        if not isinstance(ref, str):
            raise ValueError("Expected an artifact-bound event_ref")
        digest, number = ref.rsplit(":", 1)
        if digest != sha or not number.isdecimal() or int(number) not in ops:
            raise ValueError("Stale or invalid operation event_ref")
        return int(number)

    reviews = mapping.get("reviews")
    if not isinstance(reviews, list):
        raise ValueError("Alignment map reviews must be a list")
    latest = {}
    required = {"left_ref", "right_ref", "left_tensor", "right_tensor", "decision", "reason", "evidence"}
    for review in reviews:
        if (not isinstance(review, dict) or not required <= review.keys()
                or review.keys() - required - {"id", "created_at"}):
            raise ValueError("Invalid review fields; implicit transforms are unsupported")
        a, b = index(review["left_ref"]), index(review["right_ref"])
        left, right = ops[a]["args"], ops[b]["args"]
        if not left.get("run") or not right.get("run") or left["run"] == right["run"]:
            raise ValueError("A review must relate different runs")
        if review["decision"] not in DECISIONS or not isinstance(review["reason"], str) or not review["reason"].strip():
            raise ValueError("A review requires a decision and a reason")
        evidence = review["evidence"]
        if not isinstance(evidence, list) or not evidence:
            raise ValueError("A review requires evidence references")
        for item in evidence:
            if (not isinstance(item, dict) or set(item) != {"basis", "reference"}
                    or item["basis"] not in BASES or not isinstance(item["reference"], str)
                    or not item["reference"].strip()):
                raise ValueError("Invalid evidence basis/reference")
        lp, rp = review["left_tensor"], review["right_tensor"]
        if (lp is None) != (rp is None):
            raise ValueError("Select both tensor slots or neither")
        if lp is not None:
            if review["decision"] != "comparable":
                raise ValueError("Negative/undetermined reviews address the whole proposed relation; use null slots")
            tensor_slot(left, lp)
            tensor_slot(right, rp)
        elif review["decision"] == "comparable":
            if ((left.get("correspondence") or {}).get("state") != "paired"
                    or (right.get("correspondence") or {}).get("state") != "paired"
                    or (left.get("correspondence") or {}).get("peer_record_id") != right["record_id"]
                    or (right.get("correspondence") or {}).get("peer_record_id") != left["record_id"]):
                raise ValueError("New correspondences require explicit tensor slots")
        latest[tuple(sorted((a, b)))] = (a, b, review)
    used = set()
    for a, b, review in latest.values():
        if review["decision"] == "comparable":
            if a in used or b in used:
                raise ValueError("Conflicting comparable reviews: one relation per operation per view")
            used.update((a, b))
    return ops, list(latest.values())


def apply_reviews(doc, sha, mapping):
    doc = copy.deepcopy(doc)
    ops, reviews = validate_reviews(doc, sha, mapping)
    by_id = {event["args"]["record_id"]: i for i, event in ops.items()}

    def peer(i):
        args = ops[i]["args"]
        claim = args.get("correspondence") or {}
        j = by_id.get(claim.get("peer_record_id"))
        if claim.get("state") == "paired" and j is not None:
            back = ops[j]["args"].get("correspondence") or {}
            if back.get("state") == "paired" and back.get("peer_record_id") == args["record_id"]:
                return j
        return None

    def preserve(i):
        args = ops[i]["args"]
        args.setdefault("original_comparison", {k: copy.deepcopy(args.get(k))
                                               for k in ("status", "correspondence", "diff")})

    def detach(i, reason):
        mate = peer(i)
        for j in (i, mate) if mate is not None else (i,):
            preserve(j)
            args = ops[j]["args"]
            args.update(status="unaligned", correspondence={"state": "undetermined", "reason": reason,
                        "validation": "agent_reviewed", "is_data_dependency": False})
            args.pop("diff", None)

    # Save whole-event observations before replacing any candidate edges.
    originals = {i: copy.deepcopy(e["args"]) for i, e in ops.items()}
    for a, b, review in reviews:
        for i in (a, b):
            ops[i]["args"].setdefault("alignment_reviews", []).append(copy.deepcopy(review))
        if review["decision"] != "comparable" and peer(a) == b:
            detach(a, "review_" + review["decision"])
    for a, b, review in reviews:
        if review["decision"] != "comparable":
            continue
        detach(a, "superseded_by_explicit_alignment")
        detach(b, "superseded_by_explicit_alignment")
        if review["left_tensor"] is None:
            status, diff = originals[a]["status"], originals[a].get("diff")
        else:
            left = tensor_slot(originals[a], review["left_tensor"])
            right = tensor_slot(originals[b], review["right_tensor"])
            fields = [k for k in sorted(left.keys() | right.keys()) if left.get(k) != right.get(k)]
            status = "diff" if fields else "ok"
            diff = {"fields": fields, "summary": "; ".join(f"{k}: {left.get(k)!r} vs {right.get(k)!r}" for k in fields)} if fields else None
        for i, j, slot, other_slot in ((a, b, review["left_tensor"], review["right_tensor"]),
                                        (b, a, review["right_tensor"], review["left_tensor"])):
            args = ops[i]["args"]
            args.update(status=status, correspondence={"state": "paired", "kind": "reviewed_relation",
                        "basis": "explicit_alignment_map", "validation": "agent_reviewed",
                        "semantic_equivalence": "asserted_not_independently_verified",
                        "peer_record_id": ops[j]["args"]["record_id"], "is_data_dependency": False,
                        "tensor_slot": slot, "peer_tensor_slot": other_slot})
            args["comparison_scope"] = "selected_tensor_summary" if slot is not None else "original_candidate_observations"
            if diff:
                args["diff"] = copy.deepcopy(diff)
            else:
                args.pop("diff", None)

    # Rebuild visual annotations, retaining every operation and source/stack event.
    events = [e for e in doc["traceEvents"] if e.get("cat") not in ("opguard:align", "opguard:status")
              and not (e.get("ph") == "i" and e.get("args", {}).get("first_diff"))]
    first, flows = set(), set()
    for i, event in sorted(ops.items(), key=lambda item: (item[1].get("ts", 0), item[0])):
        args = event["args"]
        args["parent_event_ref"] = f"{sha}:{i}"
        args["tensor_equality"] = "not_established"
        args["comparison_evidence"] = "recorded_metadata_and_fingerprints"
        args.pop("first_diff", None)
        claim = args.setdefault("correspondence", {"state": "undetermined"})
        claim.setdefault("validation", "unreviewed")
        event["cname"] = {"ok": "good", "diff": "bad", "unaligned": "olive"}.get(args["status"], "yellow")
        args["__search"] = f"run:{args.get('run')} rank:{args.get('rank')} status:{args['status']}"
        group = (args.get("run"), args.get("rank"))
        if args["status"] == "diff" and group not in first:
            first.add(group)
            args["first_diff"] = True
            events.append({"name": "first_observed_diff", "cat": "opguard", "ph": "i", "s": "t",
                           "ts": event["ts"], "pid": event["pid"], "tid": event["tid"],
                           "args": {"first_diff": True, "run": group[0], "rank": group[1],
                                    "scope": "per_run_rank_export_order", "root_cause": "not_determined"}})
        label = {"ok": "RECORDED FIELDS AGREE", "unaligned": "UNDETERMINED",
                 "diff": "DIFF · AGENT REVIEWED" if claim.get("validation") == "agent_reviewed" else "CANDIDATE DIFF"}.get(args["status"], args["status"].upper())
        events.append({"name": label, "cat": "opguard:status", "ph": "X",
                       "ts": event["ts"], "dur": event["dur"], "pid": event["pid"],
                       "tid": f"gpu_{args.get('rank')}:status"})
        j = peer(i)
        if j is not None and tuple(sorted((i, j))) not in flows:
            flows.add(tuple(sorted((i, j))))
            endpoints = sorted((event, ops[j]), key=lambda e: e["ts"] + e["dur"] // 2)
            for phase, endpoint in zip(("s", "f"), endpoints):
                events.append({"ph": phase, "ts": endpoint["ts"] + endpoint["dur"] // 2,
                               "pid": endpoint["pid"], "tid": endpoint["tid"], "id": len(flows),
                               "cat": "opguard:align", "name": "correspondence", "bp": "e" if phase == "s" else "s"})
    doc["traceEvents"] = events
    doc.setdefault("alignment", {})["review_method"] = "explicit_alignment_map_v1"
    doc["alignment_review"] = {"artifact_sha256": sha, "map": mapping,
                               "semantic_validation": "agent_assertions_with_evidence_references",
                               "numeric_verdict": "not_evaluated", "transforms": "none"}
    return doc


def render_alignment(artifact, alignment_map, output):
    doc, sha = read_export(artifact)
    with Path(alignment_map).open("rb") as stream:
        map_bytes = stream.read(8 * 1024 * 1024 + 1)
    if len(map_bytes) > 8 * 1024 * 1024:
        raise ValueError("Alignment map exceeds the 8 MiB limit")
    mapping = json.loads(map_bytes)
    reviewed = apply_reviews(doc, sha, mapping)
    reviewed["alignment_review"]["map_sha256"] = hashlib.sha256(map_bytes).hexdigest()
    target = Path(output).expanduser().resolve()
    raw = json.dumps(reviewed, ensure_ascii=False).encode()
    if len(raw) > MAX_BYTES:
        raise ValueError("Reviewed trace exceeds the decoded size limit")
    encoded = gzip.compress(raw, mtime=0) if target.suffix == ".gz" else raw
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(encoded)
    return {"artifact": str(target), "sha256": hashlib.sha256(encoded).hexdigest(),
            "source_sha256": sha, "map_sha256": hashlib.sha256(map_bytes).hexdigest(),
            "verdict": "not_evaluated"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--alignment-map", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(render_alignment(args.artifact, args.alignment_map, args.out)))


if __name__ == "__main__":
    main()

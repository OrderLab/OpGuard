import os
import tempfile

from opguard.callsite_tracer.trace import CallsiteTracer


def _tracer(tmp, **overrides):
    t = CallsiteTracer(
        dump_every=0,
        dump_path=os.path.join(tmp, "callsite.json"),
        enable_cudaapi=False,
        enable_cupti=False,
        enable_python=False,
    )
    for k, v in overrides.items():
        setattr(t, k, v)
    return t


def test_stacks_per_site_and_frame_caps():
    with tempfile.TemporaryDirectory() as tmp:
        tracer = _tracer(tmp, max_frames=2, max_stacks_per_site=2, max_sites=0, max_total_stacks=0)
        leaf = ("leaf.py", 9, "leaf_fn")
        frames_a = [("a.py", 1, "fa"), ("b.py", 2, "fb"), leaf]
        frames_b = [("x.py", 1, "fx"), ("y.py", 2, "fy"), leaf]
        frames_c = [("p.py", 1, "fp"), ("q.py", 2, "fq"), leaf]
        events = [
            {"kernel": "k", "stack": frames_a},
            {"kernel": "k", "stack": frames_b},
            {"kernel": "k", "stack": frames_c},
            {"kernel": "k", "stack": frames_a},
        ]
        tracer._ingest_event_list(events, src="cudaapi")
        assert len(tracer.stats["cudaapi"]) == 1
        site = next(iter(tracer.stats["cudaapi"].values()))
        assert site["count"] == 4
        assert len(site["callstacks"]) == 2
        for stack in site["callstacks"]:
            assert len(stack) == 2
            assert stack[-1] == leaf


def test_site_and_total_stack_caps():
    with tempfile.TemporaryDirectory() as tmp:
        tracer = _tracer(tmp, max_frames=0, max_stacks_per_site=8, max_sites=1, max_total_stacks=1)
        tracer._ingest_event_list(
            [
                {"kernel": "k1", "stack": [("outer.py", 1, "outer"), ("a.py", 1, "fa")]},
                {"kernel": "k2", "stack": [("b.py", 2, "fb")]},
                {"kernel": "k1", "stack": [("other.py", 3, "other"), ("a.py", 1, "fa")]},
            ],
            src="python",
        )
        assert len(tracer.stats["python"]) == 1
        site = next(iter(tracer.stats["python"].values()))
        assert site["count"] == 2
        assert len(site["callstacks"]) == 1
        assert "sites" in tracer._limit_hits
        assert "total_stacks" in tracer._limit_hits

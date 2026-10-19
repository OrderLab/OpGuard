#!/usr/bin/env python3
import json
import sys
from typing import Dict, List, Tuple, Set

Frame = Tuple[str, int, str]          # (file, line, func)
Stack = Tuple[Frame, ...]              # ordered frames
StacksByKey = Dict[str, Set[Stack]]    # key -> set of stacks


def load_callstacks(path: str) -> StacksByKey:
    """
    Load a callsite JSON file and return a mapping:
      key -> set of normalized callstacks (each callstack is a tuple of frames).
    Only the 'callstacks' field is used; other fields are ignored.
    """
    with open(path, "r") as f:
        data = json.load(f)

    result: StacksByKey = {}
    for key, payload in data.items():
        stacks_raw: List[List[List]] = payload.get("callstacks", [])
        norm_stacks: Set[Stack] = set()
        for stack in stacks_raw:
            # Each stack is a list of frames: [file, line, func]
            try:
                frames: List[Frame] = []
                for fr in stack:
                    # Defensive normalize: allow tuples/lists, ensure shapes
                    if not isinstance(fr, (list, tuple)) or len(fr) != 3:
                        # Skip malformed frames quietly
                        continue
                    fpath = str(fr[0]) if fr[0] is not None else ""
                    line = int(fr[1]) if fr[1] is not None else -1
                    func = str(fr[2]) if fr[2] is not None else ""
                    frames.append((fpath, line, func))
                norm_stacks.add(tuple(frames))
            except Exception:
                # Skip malformed stacks
                continue
        result[key] = norm_stacks
    return result


def pretty_frame(fr: Frame) -> str:
    f, ln, fn = fr
    return f"{f}:{ln} in {fn}"


def pretty_stack(st: Stack) -> List[str]:
    # Show from root->leaf exactly as stored
    return [pretty_frame(fr) for fr in st]


def compare(a: StacksByKey, b: StacksByKey) -> dict:
    keys_a = set(a.keys())
    keys_b = set(b.keys())

    only_in_a = sorted(keys_a - keys_b)
    only_in_b = sorted(keys_b - keys_a)
    in_both = sorted(keys_a & keys_b)

    per_key = {}
    for k in in_both:
        a_stacks = a.get(k, set())
        b_stacks = b.get(k, set())
        new_in_b = sorted(list(b_stacks - a_stacks), key=lambda s: (len(s), s))
        missing_in_b = sorted(list(a_stacks - b_stacks), key=lambda s: (len(s), s))

        per_key[k] = {
            "new_in_B": [pretty_stack(st) for st in new_in_b],
            "missing_in_B": [pretty_stack(st) for st in missing_in_b],
            "A_count": len(a_stacks),
            "B_count": len(b_stacks),
        }

    return {
        "keys_only_in_A": only_in_a,
        "keys_only_in_B": only_in_b,
        "keys_in_both": in_both,
        "per_key_diff": per_key,
    }


def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_callsites.py <fileA.json> <fileB.json>", file=sys.stderr)
        sys.exit(2)

    file_a, file_b = sys.argv[1], sys.argv[2]
    a = load_callstacks(file_a)
    b = load_callstacks(file_b)
    diff = compare(a, b)

    # Print JSON for easy piping/processing
    print(json.dumps(diff, indent=2))


if __name__ == "__main__":
    main()

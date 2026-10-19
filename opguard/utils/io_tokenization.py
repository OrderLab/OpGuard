from collections import Counter
import hashlib

# --- utilities ---

def _shape_norm(s):
    if isinstance(s, list):
        ss = list(s)
        while ss and ss[-1] == 1:
            ss.pop()
        return tuple(ss)
    return s

def _safe_hash(s: str, n=8):
    try:
        return hashlib.blake2b(s.encode("utf-8"), digest_size=6).hexdigest()
    except Exception:
        return "?"

def _tensor_sig(obj: dict, ignored: set[str], base_path: str):
    """Return a canonical (kind, shape, numel, dtype) tuple or None."""
    # treat dicts like {"shape":[...], "stats":{"numel":..,"dtype":..,"device":..,"xor_signature":..}}
    # (common in your logs as outputs/stats)
    shape = obj.get("shape")
    numel = obj.get("stats", {}).get("numel") if isinstance(obj.get("stats"), dict) else None
    dtype = obj.get("stats", {}).get("dtype") if isinstance(obj.get("stats"), dict) else None

    # respect ignored paths
    if any(p == f"{base_path}.stats.xor_signature" or p == f"{base_path}.stats.device" for p in ignored):
        pass  # nothing to do; we just don't use those anyway

    if shape is not None and numel is not None and dtype is not None:
        return ("tensor",
                _shape_norm(shape),
                int(numel) if isinstance(numel, (int, float)) else numel,
                dtype)
    return None

def _value_token(v, ignored: set[str], base_path: str):
    """
    Turn an arbitrary value into a comparison token:
      - tensors -> ('tensor', shape, numel, dtype)
      - scalars -> ('scalar', type, value)
      - short repr for unknowns -> ('obj', type, hash(repr))
    """
    # Tensor-like dict?
    if isinstance(v, dict):
        ts = _tensor_sig(v, ignored, base_path)
        if ts is not None:
            return ts
        # structural token for dict without tensor stats
        # (we'll be more specific when walking dict key-values)
        return ("dict",)

    # simple scalars
    if isinstance(v, (str, int, float, bool, type(None))):
        # avoid huge strings; hash them
        sval = v if isinstance(v, (int, float, bool, type(None))) else _safe_hash(str(v))[:8]
        return ("scalar", type(v).__name__, sval)

    # lists will be handled by callers iterating through them; here just type
    if isinstance(v, list):
        return ("list",)

    # fallback: object type + short repr hash
    return ("obj", type(v).__name__, _safe_hash(repr(v))[:8])

def _walk_dict_tokens(d: dict, ignored: set[str], base_path: str) -> Counter:
    """
    Extract tokens from a dict treating keys as *hints* for identity (order-invariant).
    """
    bag = Counter()
    for k, v in d.items():
        pathk = f"{base_path}.{k}" if base_path else k
        # skip ignored paths (prefix-aware)
        if any(pathk == p or pathk.startswith(p + ".") for p in ignored):
            continue

        # Prefer tensor signature for leaf dicts with stats
        if isinstance(v, dict):
            ts = _tensor_sig(v, ignored, pathk)
            if ts is not None:
                # include key-name hint so two tensors swapped across keys don't falsely match
                bag[("kv-tensor", k, ts[1], ts[2], ts[3])] += 1
                continue

        tok = _value_token(v, ignored, pathk)

        # If simple scalar under a key, include the key name
        if tok and tok[0] == "scalar":
            bag[("kv-scalar", k, tok[1], tok[2])] += 1
        elif tok:
            bag[("kv", k, tok)] += 1

        # Recurse for nested dicts/lists to pick deeper tensors
        if isinstance(v, dict):
            bag += _walk_dict_tokens(v, ignored, pathk)
        elif isinstance(v, list):
            bag += _walk_list_tokens(v, ignored, f"{pathk}[]")
    return bag

def _walk_list_tokens(lst: list, ignored: set[str], base_path: str) -> Counter:
    """
    Extract tokens from a list in an order-robust way:
      - prefer tensor signatures
      - otherwise type+repr-hash tokens
    """
    bag = Counter()
    for idx, v in enumerate(lst):
        pathi = f"{base_path}.{idx}"
        if any(pathi == p or pathi.startswith(p + ".") for p in ignored):
            continue

        if isinstance(v, dict):
            ts = _tensor_sig(v, ignored, pathi)
            if ts is not None:
                bag[("tensor", ts[1], ts[2], ts[3])] += 1
            else:
                bag += _walk_dict_tokens(v, ignored, pathi)
        elif isinstance(v, list):
            bag += _walk_list_tokens(v, ignored, f"{pathi}[]")
        else:
            tok = _value_token(v, ignored, pathi)
            bag[("val", tok)] += 1
    return bag

def _bag_for_io(io_obj, ignored: set[str], base_path: str) -> Counter:
    """
    Normalize any IO (dict/list/singleton) into a multiset of tokens.
    """
    if io_obj is None:
        return Counter()
    if isinstance(io_obj, dict):
        return _walk_dict_tokens(io_obj, ignored, base_path)
    if isinstance(io_obj, list):
        return _walk_list_tokens(io_obj, ignored, base_path)
    # singleton
    tok = _value_token(io_obj, ignored, base_path)
    return Counter({("single", tok): 1})

def _soft_eq_tensor_tokens(tokA, tokB):
    """
    Allow small leniency for numel (+/-1%) and for shape after normalization.
    tok looks like ("tensor", shape, numel, dtype) or ("kv-tensor", key, shape, numel, dtype)
    """
    if tokA[0] != tokB[0]:
        return False

    # align fields regardless of kv/list form
    if tokA[0] == "tensor":
        _, sA, nA, dA = tokA
        _, sB, nB, dB = tokB
    else:
        _, kA, sA, nA, dA = tokA
        _, kB, sB, nB, dB = tokB
        # prefer matching same key if present
        if kA != kB:
            return False

    if dA != dB:
        return False

    if sA != sB:
        return False

    try:
        if nA == nB:
            return True
        diff = abs(float(nA) - float(nB)) / max(float(nA), float(nB))
        return diff <= 0.01
    except Exception:
        return False

def bag_similarity(bagA: Counter, bagB: Counter) -> float:
    """
    Jaccard-like similarity with soft matching for tensors.
    """
    # exact token intersection
    common_exact = sum((bagA & bagB).values())
    total = sum(bagA.values()) + sum(bagB.values()) - common_exact

    # attempt soft matches for tensor tokens that didn't match exactly
    def is_tensor_tok(t):
        return t and (t[0] == "tensor" or t[0] == "kv-tensor")

    remA = Counter({t: c for t, c in (bagA - (bagA & bagB)).items()})
    remB = Counter({t: c for t, c in (bagB - (bagA & bagB)).items()})

    soft_matches = 0
    # naive O(n*m) is fine for small IO bags; if large, optimize with bucketing by dtype/shape
    for tA, cA in list(remA.items()):
        if not is_tensor_tok(tA):
            continue
        for tB, cB in list(remB.items()):
            if not is_tensor_tok(tB):
                continue
            if _soft_eq_tensor_tokens(tA, tB):
                m = min(cA, cB)
                soft_matches += m
                remA[tA] -= m
                remB[tB] -= m
                if remA[tA] <= 0:
                    del remA[tA]
                if remB[tB] <= 0:
                    del remB[tB]
                break  # move next tA

    # Jaccard with soft matches counted as intersection
    inter = common_exact + soft_matches
    denom = total  # already excludes common_exact once
    if denom <= 0:
        return 1.0
    return inter / denom

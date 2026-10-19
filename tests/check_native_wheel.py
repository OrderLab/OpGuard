"""Check native release archive contents before running GPU validation."""
import json
import sys
import zipfile


with zipfile.ZipFile(sys.argv[1]) as wheel:
    names = wheel.namelist()
    for prefix in ("xor_reduce_cuda.", "opguard/callsite_tracer/opguard_cupti."):
        assert any(name.startswith(prefix) and name.endswith(".so") for name in names), prefix
    for name in ("opguard/callsite_tracer/native/libpylaunch_hook.so", "opguard/config.yaml",
                 "opguard/collections/default_config.yaml", "opguard/tracer/xor_signature.py"):
        assert name in names, name
    build = json.loads(wheel.read("opguard/_native_build.json"))
    assert build["extensions"] == {"xor": True, "callsite": True}, build
    print(json.dumps({"wheel": sys.argv[1], "native_build": build, "contents_verified": True}, indent=2))

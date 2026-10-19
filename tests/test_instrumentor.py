import ast
import tempfile
import textwrap
import json
from opguard.instrumentor.op_wrapper.file_api import instrument_source_code_robust

def normalize(code: str) -> str:
    return ast.dump(ast.parse(code), annotate_fields=False)

def test_tagged_match_instruments_call():
    input_code = textwrap.dedent("""
    class FakeModel:
        def forward(self, x):
            return rmsnorm_backward(grad_ln1, invvar, input_ln1)
    """)

    config = [{
        "function": "rmsnorm_backward",
        "class": "FakeModel",
        "method": "forward",
        "tag_names": ["ln1"],
        "original_module": "model"
    }]

    expected_code = textwrap.dedent("""
    from opguard.wrapped_ops import make_wrapper
    wrapped_ln1_rmsnorm_backward = make_wrapper()

    class FakeModel:
        def forward(self, x):
            return wrapped_ln1_rmsnorm_backward(rmsnorm_backward, grad_ln1, invvar, input_ln1)
    """)

    with tempfile.NamedTemporaryFile(delete=False) as f:
        log_path = f.name

    output_code = instrument_source_code(input_code, config, log_path=log_path)
    assert normalize(output_code) == normalize(expected_code)

def test_unconditional_wrapping_when_no_tags():
    input_code = textwrap.dedent("""
    class FakeModel:
        def forward(self, x):
            return rmsnorm_backward(grad, invvar, input)
    """)

    config = [{
        "function": "rmsnorm_backward",
        "class": "FakeModel",
        "method": "forward",
        "original_module": "model"
    }]

    expected_code = textwrap.dedent("""
    from opguard.wrapped_ops import make_wrapper
    wrapped_rmsnorm_backward = make_wrapper()

    class FakeModel:
        def forward(self, x):
            return wrapped_rmsnorm_backward(rmsnorm_backward, grad, invvar, input)
    """)

    with tempfile.NamedTemporaryFile(delete=False) as f:
        log_path = f.name

    output_code = instrument_source_code(input_code, config, log_path=log_path)
    assert normalize(output_code) == normalize(expected_code)

def test_branching_calls_are_both_instrumented():
    input_code = textwrap.dedent("""
    class TransformerLayer:
        def backward(self, condition, mlp_output_parallel, hidden_size, lnf_weight, eps):
            if condition:
                mlp_output, lnf_invvar = rmsnorm_forward(mlp_output_parallel, hidden_size, lnf_weight, eps)
            else:
                mlp_output, lnf_invvar = rmsnorm_forward(mlp_output_parallel, hidden_size, lnf_weight, eps)
    """)

    config = [{
        "function": "rmsnorm_forward",
        "class": "TransformerLayer",
        "method": "backward",
        "tag_names": ["mlp_output"],
        "original_module": "model"
    }]

    with tempfile.NamedTemporaryFile(delete=False) as f:
        log_path = f.name

    output_code = instrument_source_code(input_code, config, log_path=log_path)

    # Only count function call usages, not declarations or assignments
    actual_calls = [
        line.strip() for line in output_code.splitlines()
        if "wrapped_mlp_output_rmsnorm_forward(" in line
        and "make_wrapper" not in line
    ]

    assert len(actual_calls) == 2, f"Expected 2 wrapped calls, found {len(actual_calls)}.\nOutput:\n{output_code}"

    # Optional: validate log entries
    with open(log_path) as f:
        wrapped_log = json.load(f)
    assert sum(e["wrapped_name"] == "wrapped_mlp_output_rmsnorm_forward" for e in wrapped_log) == 2

def test_extract_var_name_from_chained_call():
    input_code = textwrap.dedent("""
    class TransformerLayer:
        def forward(self, condition, mlp_output_parallel, hidden_size, lnf_weight, eps):
            if condition:
                mlp_output, lnf_invvar = rmsnorm_forward(mlp_output_parallel, hidden_size, lnf_weight, eps)
            else:
                mlp_output, lnf_invvar = rmsnorm_forward(mlp_output_parallel.float(), hidden_size, lnf_weight, eps)
    """)

    config = [{
        "function": "rmsnorm_forward",
        "class": "TransformerLayer",
        "method": "forward",
        "tag_names": ["mlp_output"],
        "original_module": "model"
    }]

    with tempfile.NamedTemporaryFile(delete=False) as f:
        log_path = f.name

    output_code = instrument_source_code(input_code, config, log_path=log_path)

    # Count only the wrapped function *calls*
    actual_calls = [
        line.strip() for line in output_code.splitlines()
        if "wrapped_mlp_output_rmsnorm_forward(" in line
        and "make_wrapper" not in line
    ]

    assert len(actual_calls) == 2, f"Expected 2 wrapped calls (if + else), found {len(actual_calls)}.\nOutput:\n{output_code}"

    with open(log_path) as f:
        wrapped_log = json.load(f)

    assert sum(e["wrapped_name"] == "wrapped_mlp_output_rmsnorm_forward" for e in wrapped_log) == 2
import unittest
import libcst as cst
from libcst.metadata import MetadataWrapper
from opguard.instrumentor.meta_var_injector import MetaVarInjector  # replace with your actual import

class TestMetaVarInjector(unittest.TestCase):
    def test_forward_and_backward_injection(self):
        source_code = """
class DummyModule:
    def forward(self, hidden_states, micro_batch_idx, global_step):
        hidden_states = hidden_states + 1
        return hidden_states, {}
"""
        # Wrap source as CST
        module = cst.parse_module(source_code)

        # Configure injection to instrument micro_batch_idx and global_step
        target_config = {
            "class": "DummyModule",
            "function": "forward",
            "var_name": ["micro_batch_idx", "global_step"]
        }

        wrapper = MetadataWrapper(module)
        transformed = wrapper.visit(MetaVarInjector(target_config))

        transformed_code = transformed.code
        print("[Transformed code]\n", transformed_code)

        transformed_code_no_spaces = transformed_code.replace(" ", "")
        self.assertIn("update_meta(micro_batch_idx=micro_batch_idx)", transformed_code_no_spaces)
        self.assertIn("update_meta(global_step=global_step)", transformed_code_no_spaces)
        self.assertIn(
            "hidden_states.register_hook(lambdagrad:update_meta(micro_batch_idx=micro_batch_idx,global_step=global_step)orgrad)",
            transformed_code_no_spaces
        )


if __name__ == "__main__":
    unittest.main()

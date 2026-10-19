#!/usr/bin/env python3
import sys, argparse
import libcst as cst
from libcst.metadata import PositionProvider, MetadataWrapper

class InsertLineno(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (PositionProvider,)

    def _is_wrapped_name(self, node: cst.BaseExpression) -> bool:
        return isinstance(node, cst.Name) and node.value.startswith("opguard_wrapped_")

    def _find_wrapped_name(self, expr: cst.BaseExpression):
        cur = expr
        # Walk through nested Call(...).func until we hit Name("opguard_wrapped_...")
        while isinstance(cur, cst.Call):
            if self._is_wrapped_name(cur.func):
                return cur.func
            cur = cur.func
        return expr if self._is_wrapped_name(expr) else None

    def _strip_leading_ints(self, args):
        i = 0
        while i < len(args) and isinstance(args[i].value, cst.Integer):
            i += 1
        return list(args[i:])

    def leave_Call(self, original: cst.Call, updated: cst.Call) -> cst.CSTNode:
        wrapped_name = self._find_wrapped_name(updated.func)
        if wrapped_name is None:
            return updated

        # Current callsite line (stable on reruns when we don't mutate first)
        lineno = self.get_metadata(PositionProvider, original).start.line
        lineno_arg = cst.Arg(value=cst.Integer(str(lineno)))

        # Canonical form: opguard_wrapped_*(lineno)(<real args...>)
        new_func = cst.Call(func=wrapped_name, args=[lineno_arg])

        # If previous runs leaked linenos into the OUTER arg list, drop them
        new_args = self._strip_leading_ints(updated.args)

        return updated.with_changes(func=new_func, args=new_args)

def apply_rewrite(in_path: str, out_path: str) -> None:
    try:
        with open(in_path, "r", encoding="utf-8") as f:
            source = f.read()
        tree = MetadataWrapper(cst.parse_module(source)).visit(InsertLineno())
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(tree.code)
    except Exception as e:
        print(f"Error writing to {out_path}: {e}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("input_file")
    p.add_argument("-o", "--output-file")
    args = p.parse_args()
    apply_rewrite(args.input_file, args.output_file or args.input_file)

if __name__ == "__main__":
    main()

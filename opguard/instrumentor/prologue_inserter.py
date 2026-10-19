# --- new Phase-B inserter ---
import os
import libcst as cst

class PrologueInserter(cst.CSTTransformer):
    def __init__(self, wrapped_imports, need_operator_import, need_flush_import, defer_prologue=False):
        # wrapped_imports: list of (wrapped_name, original_name_or_dotted, tag_prefix, tensor_dump, global_sync)
        self.wrapped_imports = wrapped_imports
        self.need_operator_import = need_operator_import
        self.need_flush_import = need_flush_import
        self.defer_prologue = defer_prologue

    def _attr_to_str(self, node: cst.BaseExpression) -> str:
        if isinstance(node, cst.Name):
            return node.value
        if isinstance(node, cst.Attribute):
            return f"{self._attr_to_str(node.value)}.{node.attr.value}"
        return ""

    def _module_has_import_name(self, mod: cst.Module, name: str) -> bool:
        for s in mod.body:
            if isinstance(s, cst.SimpleStatementLine):
                for small in s.body:
                    if isinstance(small, cst.Import):
                        for alias in small.names:
                            if (isinstance(alias, cst.ImportAlias)
                                and isinstance(alias.name, cst.Name)
                                and alias.name.value == name):
                                return True
        return False

    def _module_has_importfrom(self, mod: cst.Module, module_path: str, obj_name: str) -> bool:
        for s in mod.body:
            if isinstance(s, cst.SimpleStatementLine):
                for small in s.body:
                    if isinstance(small, cst.ImportFrom) and small.module is not None:
                        if self._attr_to_str(small.module) == module_path:
                            for alias in small.names:
                                if (isinstance(alias, cst.ImportAlias)
                                    and isinstance(alias.name, cst.Name)
                                    and alias.name.value == obj_name):
                                    return True
        return False

    def _module_has_assignment_to(self, mod: cst.Module, target_name: str) -> bool:
        for s in mod.body:
            if isinstance(s, cst.SimpleStatementLine):
                for small in s.body:
                    if isinstance(small, cst.Assign):
                        for tgt in small.targets:
                            if isinstance(tgt.target, cst.Name) and tgt.target.value == target_name:
                                return True
        return False

    def leave_Module(self, original: cst.Module, updated: cst.Module) -> cst.Module:
        # Deduplicate wrapper assignments requested during Phase A
        unique_assignments = set()
        for wrapped_name, orig_name, tag_prefix, tensor_dump, global_sync in self.wrapped_imports:
            canon_key = (
                str(wrapped_name),
                str(orig_name) if orig_name else "",
                str(tag_prefix),
                bool(tensor_dump),
                bool(global_sync),
            )
            unique_assignments.add(canon_key)

        assigns_to_insert = []
        for wrapped_name, orig_name, tag_prefix, tensor_dump, global_sync in sorted(unique_assignments):
            if self._module_has_assignment_to(updated, wrapped_name):
                continue
            kw = [
                cst.Arg(keyword=cst.Name("tag_prefix"), value=cst.SimpleString(repr(tag_prefix))),
            ]
            if tensor_dump:
                kw.append(cst.Arg(keyword=cst.Name("dump_tensors"), value=cst.Name("True")))
            if global_sync:
                kw.append(cst.Arg(keyword=cst.Name("global_sync"), value=cst.Name("True")))
            assigns_to_insert.append(
                cst.SimpleStatementLine([
                    cst.Assign(
                        targets=[cst.AssignTarget(target=cst.Name(wrapped_name))],
                        value=cst.Call(func=cst.Name("make_wrapper"), args=kw)
                    )
                ])
            )

        # imports (guarded, using same conditions as before)
        stmts_to_prepend = []
        have_operator = self._module_has_import_name(updated, "operator")
        have_makewrap = self._module_has_importfrom(updated, "opguard.wrapped_ops", "make_wrapper")
        have_flushimp = self._module_has_importfrom(updated, "opguard.runtime_tracer", "flush_trace_buffer")

        if self.need_flush_import and not have_flushimp:
            stmts_to_prepend.append(
                cst.SimpleStatementLine([
                    cst.ImportFrom(
                        module=cst.Attribute(value=cst.Name("opguard"), attr=cst.Name("runtime_tracer")),
                        names=[cst.ImportAlias(name=cst.Name("flush_trace_buffer"))],
                        relative=[]
                    )
                ])
            )

        if assigns_to_insert and not have_makewrap:
            stmts_to_prepend.append(
                cst.SimpleStatementLine([
                    cst.ImportFrom(
                        module=cst.Attribute(value=cst.Name("opguard"), attr=cst.Name("wrapped_ops")),
                        names=[cst.ImportAlias(name=cst.Name("make_wrapper"))],
                        relative=[]
                    )
                ])
            )

        if (self.need_operator_import or any("operator." in (s or "") for _, s, *_ in unique_assignments)) and not have_operator:
            stmts_to_prepend.append(
                cst.SimpleStatementLine([
                    cst.Import(names=[cst.ImportAlias(name=cst.Name("operator"))])
                ])
            )

        # Determine safe insertion index: after module docstring and all __future__ imports
        body = list(updated.body)

        def _is_docstring_stmt(stmt: cst.CSTNode) -> bool:
            if not isinstance(stmt, cst.SimpleStatementLine):
                return False
            if len(stmt.body) != 1:
                return False
            small = stmt.body[0]
            return isinstance(small, cst.Expr) and isinstance(small.value, cst.SimpleString)

        def _is_future_import_stmt(stmt: cst.CSTNode) -> bool:
            if not isinstance(stmt, cst.SimpleStatementLine):
                return False
            for small in stmt.body:
                if isinstance(small, cst.ImportFrom) and small.module is not None:
                    if self._attr_to_str(small.module) == "__future__":
                        return True
            return False

        insert_idx = 0
        if body:
            if _is_docstring_stmt(body[0]):
                insert_idx = 1
            # Skip over all from __future__ imports immediately following
            while insert_idx < len(body) and _is_future_import_stmt(body[insert_idx]):
                insert_idx += 1

        # If nothing to insert, return unchanged
        if not stmts_to_prepend and not assigns_to_insert:
            return updated

        new_body = body[:insert_idx] + list(stmts_to_prepend) + list(assigns_to_insert) + body[insert_idx:]
        return updated.with_changes(body=new_body)

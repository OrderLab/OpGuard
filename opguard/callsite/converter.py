# opguard/callsite/converter.py

from __future__ import annotations
import os
import re
import json
import yaml
from typing import Dict, List, Optional, Tuple
from .callsite_parser import CallsiteParser
from opguard.instrument_skip import should_skip_instrument_path

class Quoted(str):
    pass

def _repr_quoted(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='"')

yaml.add_representer(Quoted, _repr_quoted)


def _quote_colon_keys(obj):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            kk = Quoted(str(k)) if (isinstance(k, str) and ':' in k) else k
            out[kk] = _quote_colon_keys(v)
        return out
    if isinstance(obj, list):
        return [_quote_colon_keys(x) for x in obj]
    return obj

class CallsiteToInstrumentationConverter:
    """Converts callsite trace data to instrumentation schema (refactored)."""

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or os.getcwd()
        self.parser = CallsiteParser()
        self.generated_instrumentations: List[Dict] = []
        self.seen_entries = set()
        self.missing_entries = set()
        self.invalid_entries = set()

    def get_relative_path(self, absolute_path: str) -> str:
        base = os.path.normpath(self.base_dir)
        ap   = os.path.normpath(absolute_path)
        try:
            if os.path.commonpath([base, ap]) == base:
                return os.path.relpath(ap, base)
        except Exception:
            pass
        return absolute_path

    def _context_for(self, file_path: str, line_number: int):
        return self.parser.context_for_line(file_path, line_number)

    def _seen_key(self, file_path: str, line_number: int, function_name: str,
                  col_offset: Optional[int] = None) -> str:
        # include column to disambiguate multiple ops on the same line
        if col_offset is None:
            return f"{file_path}:{line_number}:{function_name}"
        return f"{file_path}:{line_number}:{col_offset}:{function_name}"

    def process_callsite_infos(
        self,
        file_path: str,
        line_number: int,
        infos: List,  # List[CallsiteInfo]
        extra_meta: Optional[Dict] = None
    ) -> List[Dict]:
        out = []
        cls_name, meth_name = self._context_for(file_path, line_number)
        # Resolve to absolute path using base_dir for relative paths
        if os.path.isabs(file_path):
            absolute_path = os.path.normpath(file_path)
        else:
            # Resolve relative paths relative to base_dir, not current working directory
            absolute_path = os.path.normpath(os.path.join(self.base_dir, file_path))

        if should_skip_instrument_path(absolute_path):
            return out

        for info in infos:
            key = self._seen_key(file_path, line_number, info.function, getattr(info, "col_offset", None))
            if key in self.seen_entries:
                continue
            self.seen_entries.add(key)

            entry = {
                "file_path": absolute_path,
                "line_num": line_number,
                "function": info.function,
                "operation_type": info.operation_type,
            }

            # attach fine-grained location (lets the instrumentor find the exact node)
            if hasattr(info, "col_offset"):     entry["col_offset"] = info.col_offset
            if hasattr(info, "end_col_offset"): entry["end_col_offset"] = info.end_col_offset
            if hasattr(info, "end_lineno"):     entry["end_lineno"] = info.end_lineno
            if getattr(info, "snippet", None):  entry["snippet"] = info.snippet  # optional / debug

            context = []
            if cls_name:
                context.append({"type": "class", "name": cls_name})
            if meth_name:
                context.append({"type": "method", "name": meth_name})
            if context:
                entry["context"] = context

            entry["tag_names"] = info.tag_names()
            entry["tag_info"] = info.tag_info()

            if extra_meta:
                if extra_meta.get("ops"):
                    entry["ops"] = list(extra_meta["ops"])
                if extra_meta.get("stream_counts") is not None:
                    entry["stream_counts"] = extra_meta["stream_counts"]

            out.append(entry)
        return out

    def process_callsite_entry(self, file_path: str, line_number: int, function_name: str, extra_meta: Optional[Dict] = None) -> Optional[Dict]:
        """Back-compat single-entry path (unchanged externally)."""
        infos = self.parser.analyze_line(file_path, line_number)
        if not infos:
            self.missing_entries.add(f"{file_path}:{line_number}:{function_name}")
            return None
        # Keep original “winner” as the single entry
        entries = self.process_callsite_infos(file_path, line_number, [infos[0]], extra_meta=extra_meta)
        return entries[0] if entries else None

    def convert_callsite_data(self, callsite_file: str) -> List[Dict]:
        with open(callsite_file, 'r') as f:
            data = json.load(f)

        callsite_index = {}
        for k, v in data.items():
            m = re.match(r'(.+):(\d+) in (.+)', k)
            if m:
                callsite_index[(m.group(1), int(m.group(2)), m.group(3))] = v

        for callsite_key, callsite_data in data.items():
            m = re.match(r'(.+):(\d+) in (.+)', callsite_key)
            if not m:
                self.invalid_entries.add(callsite_key)
                continue

            fp = m.group(1)
            ln = int(m.group(2))
            fn = m.group(3)
            root_meta = {
                "ops": callsite_data.get("ops", []),
                "stream_counts": callsite_data.get("stream_counts"),
            }

            root_infos = self.parser.analyze_line(fp, ln)
            if root_infos:
                self.generated_instrumentations.extend(
                    self.process_callsite_infos(fp, ln, root_infos, extra_meta=root_meta)
                )
            else:
                self.missing_entries.add(f"{fp}:{ln}:{fn}")

            # Leaves
            for stack in callsite_data.get("callstacks", []) or []:
                if not stack:
                    continue
                leaf = stack[-1]
                if len(leaf) >= 3:
                    lfp, lln, lfn = leaf[0], int(leaf[1]), leaf[2]
                    leaf_meta_src = callsite_index.get((lfp, lln, lfn))
                    leaf_meta = None
                    if leaf_meta_src:
                        leaf_meta = {
                            "ops": leaf_meta_src.get("ops", []),
                            "stream_counts": leaf_meta_src.get("stream_counts"),
                        }
                    leaf_infos = self.parser.analyze_line(lfp, lln)
                    if leaf_infos:
                        self.generated_instrumentations.extend(
                            self.process_callsite_infos(lfp, lln, leaf_infos, extra_meta=leaf_meta)
                        )
                    else:
                        self.missing_entries.add(f"{lfp}:{lln}:{lfn}")

        return self.generated_instrumentations

    def generate_yaml_config(self, output_file: Optional[str] = None) -> str:
        config = {"instrumentations": self.generated_instrumentations}
        config = _quote_colon_keys(config)
        yaml_content = yaml.dump(config, default_flow_style=False, sort_keys=False)
        if output_file:
            with open(output_file, 'w') as f:
                f.write(yaml_content)
        return yaml_content

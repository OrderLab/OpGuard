"""
Source file collector for callstack-related .py files.

This module extracts file paths from callstacks, saves the source files,
and maintains a mapping for use in perfetto reports.
Also extracts function context information for highlighting.
"""
from __future__ import annotations
import os
from opguard.local_sync.paths import atomic_output
import sys
import re
import shutil
import hashlib
import ast
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Set, Optional, Tuple, List
import logging

# Pattern to match stack trace lines: "path:line in function"
_STACK_LINE_RE = re.compile(r"^(.*?):(\d+)\s+in\s+(.*)$")


def _normalize_path(path: str) -> Optional[Path]:
    """Normalize a file path and return as Path if it exists."""
    if not path or path == "<unknown>" or path == "?":
        return None
    try:
        p = Path(path).resolve()
        if p.exists() and p.is_file():
            return p
    except Exception:
        pass
    return None


def _safe_read_file(path: Path) -> Optional[str]:
    """Safely read a file, returning None on any error."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception as e:
        logging.debug(f"[OpGuard]: Failed to read source file {path}: {e}")
        return None


def _compute_file_hash(content: str) -> str:
    """Compute a short hash for file content."""
    return hashlib.md5(content.encode("utf-8")).hexdigest()[:8]


def _extract_function_context_with_tree(content: str, tree: ast.AST, line_num: int) -> Optional[Dict[str, any]]:
    """
    Extract function context using a pre-parsed AST tree.
    This avoids re-parsing the same file multiple times.
    """
    if not content or not tree:
        return None
    
    try:
        lines = content.splitlines()
        
        # Find the function containing the line (visit in order)
        class FunctionVisitor(ast.NodeVisitor):
            def __init__(self, target_line):
                self.target_line = target_line
                self.found_function = None
            
            def visit_FunctionDef(self, node):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_start = node.lineno
                    # Get end line from the last node in the function body
                    func_end = func_start
                    if node.body:
                        for stmt in node.body:
                            # Recursively find the maximum line number
                            for child in ast.walk(stmt):
                                if hasattr(child, 'lineno') and child.lineno:
                                    func_end = max(func_end, child.lineno)
                    
                    # Check if our line is within this function
                    if func_start <= self.target_line <= func_end:
                        if self.found_function is None or func_start <= self.found_function[1]:
                            # Prefer the innermost function
                            self.found_function = (node, func_start, func_end)
                
                self.generic_visit(node)
            
            def visit_AsyncFunctionDef(self, node):
                self.visit_FunctionDef(node)
        
        visitor = FunctionVisitor(line_num)
        visitor.visit(tree)
        
        if visitor.found_function:
            node, func_start, func_end = visitor.found_function
            # Find the complete function definition (may span multiple lines)
            # The function signature ends at the first ':' that's not inside parentheses
            func_def_end = func_start
            paren_count = 0
            in_string = False
            string_char = None
            
            # Parse lines starting from func_start to find where the signature ends (the ':')
            for i in range(func_start - 1, min(len(lines), func_start + 10)):  # Check up to 10 lines for the signature
                line = lines[i]
                for char in line:
                    if char in ('"', "'") and (i == func_start - 1 or lines[i-1][-1] != '\\'):
                        if not in_string:
                            in_string = True
                            string_char = char
                        elif char == string_char:
                            in_string = False
                            string_char = None
                    elif not in_string:
                        if char == '(':
                            paren_count += 1
                        elif char == ')':
                            paren_count -= 1
                        elif char == ':' and paren_count == 0:
                            func_def_end = i + 1  # i is 0-indexed, func_def_end is 1-indexed
                            break
                if func_def_end > func_start:
                    break
            
            # If we didn't find ':', assume single line definition
            if func_def_end == func_start:
                func_def_end = func_start
            
            # Always include complete function definition, then ±20 lines around target
            target_context_start = max(0, line_num - 20)
            target_context_end = min(func_end, line_num + 20)
            
            # Determine what to show
            show_func_def_separately = (target_context_start > func_def_end)  # Gap between func def and target area
            
            formatted_lines = []
            
            # Always show complete function definition (all lines from func_start to func_def_end)
            for i in range(func_start, func_def_end + 1):
                if i <= len(lines):
                    func_def_line = lines[i - 1]  # i is 1-indexed
                    if i == func_start:
                        formatted_lines.append(f"    {i:4d} | {func_def_line}  # function def")
                    else:
                        formatted_lines.append(f"    {i:4d} | {func_def_line}")
            
            # If there's a gap, show ellipsis
            if show_func_def_separately:
                skipped_lines = target_context_start - func_def_end - 1
                formatted_lines.append(f"    ... ({skipped_lines} lines skipped)")
            
            # Show ±20 lines around target (excluding func def lines if they were already shown)
            context_start = max(func_def_end + 1, target_context_start)  # Skip func def if already shown
            context_end = target_context_end
            
            start_idx = max(0, context_start - 1)  # Convert to 0-indexed
            end_idx = min(len(lines), context_end)
            context_lines = lines[start_idx:end_idx]
            
            for i, line in enumerate(context_lines, start=context_start):
                if i == line_num:
                    # Mark the target line clearly with >>> markers
                    formatted_lines.append(f">>> {i:4d} | {line} <<< TARGET LINE")
                else:
                    formatted_lines.append(f"    {i:4d} | {line}")
            
            # Add ellipsis if we truncated the end of the function
            if context_end < func_end:
                formatted_lines.append(f"    ... ({func_end - context_end} more lines)")
            
            formatted_code = '\n'.join(formatted_lines)
            
            return {
                "function_name": node.name,
                "start_line": func_start,
                "end_line": func_end,
                "target_line": line_num,
                "function_code_formatted": formatted_code,
            }
    except Exception as e:
        logging.debug(f"[OpGuard]: Failed to extract function context for line {line_num}: {e}")
    
    # Fallback: return a context around the line (±20 lines)
    lines = content.splitlines()
    if 0 <= line_num - 1 < len(lines):
        start = max(0, line_num - 21)  # 20 lines before + target line
        end = min(len(lines), line_num + 20)  # target line + 20 lines after
        context_lines = lines[start:end]
        
        formatted_lines = []
        for i, line in enumerate(context_lines, start=start + 1):
            if i == line_num:
                formatted_lines.append(f">>> {i:4d} | {line} <<< TARGET LINE")
            else:
                formatted_lines.append(f"    {i:4d} | {line}")
        
        formatted_code = '\n'.join(formatted_lines)
        
        return {
            "function_name": None,
            "start_line": start + 1,
            "end_line": end,
            "target_line": line_num,
            "function_code_formatted": formatted_code,
        }
    
    return None

def _extract_function_context(content: str, line_num: int) -> Optional[Dict[str, any]]:
    """
    Extract function context for a given line number.
    This version parses the AST each time (slower).
    For better performance, use _extract_function_context_with_tree with a cached AST.
    """
    if not content:
        return None
    
    try:
        # Parse the AST
        tree = ast.parse(content)
        return _extract_function_context_with_tree(content, tree, line_num)
    except Exception as e:
        logging.debug(f"[OpGuard]: Failed to parse AST for function context: {e}")
        return None


def _sanitize_filename(path: Path) -> str:
    """Create a safe filename from a path."""
    # Replace path separators and special chars with underscores
    parts = []
    for part in path.parts:
        parts.append(re.sub(r'[^\w\-.]', '_', part))
    return "_".join(parts)


class SourceFileCollector:
    """
    Collects and saves source files referenced in callstacks.
    
    Files are saved to a `source_files/` directory in the trace output,
    with a mapping from original paths to saved paths.
    """
    
    def __init__(self, output_base: str):
        """
        Initialize the collector.
        
        Args:
            output_base: Base directory for trace output (where source_files/ will be created)
        """
        self.output_base = Path(output_base)
        self.source_dir = self.output_base / "source_files"
        self.source_dir.mkdir(parents=True, exist_ok=True)
        
        # Mapping: original_path -> (saved_path, file_hash)
        self.file_map: Dict[str, Tuple[str, str]] = {}
        # Track which files we've already processed
        self.processed: Set[str] = set()
        # Cache file contents for function context extraction
        self.file_contents: Dict[str, str] = {}
        # Cache parsed AST trees to avoid re-parsing the same file
        self.ast_cache: Dict[str, ast.AST] = {}
        # Cache function context results: (file_path, line_num) -> context_dict
        self.function_context_cache: Dict[Tuple[str, int], Optional[Dict[str, any]]] = {}
        # Cache repeated full stack strings -> resolved file mapping.
        # Runtime stacks are often highly repetitive; this avoids re-parsing each trace.
        self.stack_trace_cache: OrderedDict[str, Dict[str, str]] = OrderedDict()
        try:
            self.stack_trace_cache_limit = int(os.getenv("OPGUARD_STACK_TRACE_CACHE_SIZE", "2048"))
        except ValueError:
            self.stack_trace_cache_limit = 2048
        if self.stack_trace_cache_limit < 0:
            self.stack_trace_cache_limit = 0
        
        # Create index file path
        self.index_file = self.source_dir / "file_index.json"
        self._load_index()

    def _stack_cache_get(self, stack_trace: str) -> Optional[Dict[str, str]]:
        """Get a cached stack->mapping result and refresh LRU order."""
        if self.stack_trace_cache_limit <= 0:
            return None
        cached = self.stack_trace_cache.get(stack_trace)
        if cached is None:
            return None
        self.stack_trace_cache.move_to_end(stack_trace)
        return dict(cached)

    def _stack_cache_put(self, stack_trace: str, file_mapping: Dict[str, str]) -> None:
        """Store stack->mapping in bounded LRU cache."""
        if self.stack_trace_cache_limit <= 0:
            return
        self.stack_trace_cache[stack_trace] = dict(file_mapping)
        self.stack_trace_cache.move_to_end(stack_trace)
        while len(self.stack_trace_cache) > self.stack_trace_cache_limit:
            self.stack_trace_cache.popitem(last=False)
    
    def _load_index(self):
        """Load existing file index if it exists."""
        if self.index_file.exists():
            try:
                import json
                with open(self.index_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.file_map = {k: tuple(v) for k, v in data.items()}
                    self.processed = set(self.file_map.keys())
            except Exception as e:
                logging.warning(f"[OpGuard]: Failed to load source file index: {e}")
    
    def _save_index(self):
        """Save the file index to disk."""
        try:
            import json
            with atomic_output(self.index_file) as temporary:
                with open(temporary, "w", encoding="utf-8") as f:
                    json.dump(
                        {k: list(v) for k, v in self.file_map.items()},
                        f,
                        indent=2,
                        ensure_ascii=False
                    )
        except Exception as e:
            logging.warning(f"[OpGuard]: Failed to save source file index: {e}")
    
    def extract_files_from_stack(self, stack_trace: Optional[str]) -> Set[str]:
        """
        Extract file paths from a stack trace string.
        
        Args:
            stack_trace: Stack trace string in format "path:line in function\\n..."
            
        Returns:
            Set of file paths found in the stack trace
        """
        files = set()
        if not stack_trace:
            return files
        
        lines = stack_trace.strip().splitlines()
        
        for idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            m = _STACK_LINE_RE.match(line)
            if m:
                file_path = m.group(1)
                if file_path and file_path not in ("<unknown>", "?"):
                    files.add(file_path)
        
        return files
    
    def save_file(self, original_path: str) -> Optional[str]:
        """
        Save a source file if it exists and is a .py file.
        
        Args:
            original_path: Original file path from stack trace
            
        Returns:
            Relative path to saved file (from source_dir), or None if not saved
        """
        if not original_path or original_path in self.processed:
            # Return existing mapping if we've seen it
            if original_path in self.file_map:
                return self.file_map[original_path][0]
            return None
        
        path = _normalize_path(original_path)
        if not path:
            self.processed.add(original_path)
            return None
        
        # Only save .py files
        if path.suffix != ".py":
            self.processed.add(original_path)
            return None
        
        # Read file content
        content = _safe_read_file(path)
        if not content:
            self.processed.add(original_path)
            return None
        
        # Cache content for function context extraction
        self.file_contents[original_path] = content
        
        # Compute hash to detect duplicates
        file_hash = _compute_file_hash(content)
        
        # Create safe filename
        safe_name = _sanitize_filename(path)
        # Include hash to handle files with same name from different locations
        saved_name = f"{safe_name}_{file_hash}.py"
        saved_path = self.source_dir / saved_name
        
        # Save file if it doesn't exist
        if not saved_path.exists():
            try:
                with atomic_output(saved_path) as temporary:
                    with open(temporary, "w", encoding="utf-8", errors="replace") as f:
                        f.write(content)
            except Exception as e:
                logging.error(f"[OpGuard]: Failed to save source file {saved_path}: {e}", exc_info=True)
                self.processed.add(original_path)
                return None
        
        # Store mapping
        rel_path = f"source_files/{saved_name}"
        self.file_map[original_path] = (rel_path, file_hash)
        self.processed.add(original_path)
        
        # Save index periodically (could be optimized with batching)
        self._save_index()
        
        return rel_path
    
    def process_stack_trace(self, stack_trace: Optional[str]) -> Dict[str, str]:
        """
        Process a stack trace, extract files, and save them.
        
        Args:
            stack_trace: Stack trace string
            
        Returns:
            Dictionary mapping original paths to saved relative paths
        """
        if not stack_trace:
            return {}

        cached = self._stack_cache_get(stack_trace)
        if cached is not None:
            return cached

        files = self.extract_files_from_stack(stack_trace)
        result = {}
        
        for file_path in files:
            saved_path = self.save_file(file_path)
            if saved_path:
                result[file_path] = saved_path

        self._stack_cache_put(stack_trace, result)
        return result
    
    def get_file_mapping(self, original_path: str) -> Optional[str]:
        """
        Get the saved path for an original file path.
        
        Args:
            original_path: Original file path
            
        Returns:
            Relative path to saved file, or None if not found
        """
        if original_path in self.file_map:
            return self.file_map[original_path][0]
        return None
    
    def get_function_context(self, original_path: str, line_num: int) -> Optional[Dict[str, any]]:
        """
        Get function context for a given file and line number.
        Uses multiple levels of caching:
        1. Function context cache: (file_path, line_num) -> result
        2. AST cache: file_path -> parsed AST tree
        3. File contents cache: file_path -> file content
        
        Args:
            original_path: Original file path
            line_num: Line number (1-indexed)
            
        Returns:
            Dict with function context, or None if not available
        """
        # Check function context cache first (fastest)
        cache_key = (original_path, line_num)
        if cache_key in self.function_context_cache:
            return self.function_context_cache[cache_key]
        
        # Try to get file content from cache
        content = self.file_contents.get(original_path)
        if not content:
            # Try to read from saved file
            saved_path = self.get_file_mapping(original_path)
            if saved_path:
                full_path = self.output_base / saved_path
                logging.debug(f"[OpGuard]: Looking up function context for {original_path}:{line_num} from {full_path}")
                if full_path.exists():
                    content = _safe_read_file(full_path)
                    if content:
                        self.file_contents[original_path] = content
                        logging.debug(f"[OpGuard]: Loaded file content from {full_path}")
                else:
                    logging.warning(f"[OpGuard]: Source file not found at expected path: {full_path} (original_path: {original_path})")
            else:
                logging.debug(f"[OpGuard]: No file mapping found for {original_path}")
        
        if content:
            # Use cached AST if available, otherwise parse and cache
            tree = self.ast_cache.get(original_path)
            if tree is None:
                try:
                    tree = ast.parse(content)
                    self.ast_cache[original_path] = tree
                except Exception as e:
                    logging.debug(f"[OpGuard]: Failed to parse AST for {original_path}: {e}")
                    result = None
                    self.function_context_cache[cache_key] = result
                    return result
            
            result = _extract_function_context_with_tree(content, tree, line_num)
            # Cache the result
            self.function_context_cache[cache_key] = result
            return result
        
        # Cache None result to avoid repeated file lookups
        self.function_context_cache[cache_key] = None
        return None
    
    def finalize(self):
        """Finalize the collector and save the index."""
        self._save_index()

    def load_from_directory(self, source_dir) -> int:
        """
        Load existing source files and index from a directory.
        
        Args:
            source_dir: Directory containing source_files/ subdirectory (Path or str)
            
        Returns:
            Number of files loaded
        """
        # Convert to Path if it's a string
        if isinstance(source_dir, str):
            source_dir = Path(source_dir)
        
        if not source_dir.exists() or not source_dir.is_dir():
            logging.debug(f"[OpGuard]: Source directory does not exist or is not a directory: {source_dir}")
            return 0
        
        source_files_dir = source_dir / "source_files"
        if not source_files_dir.exists():
            logging.debug(f"[OpGuard]: source_files directory not found: {source_files_dir}")
            return 0
        
        logging.info(f"[OpGuard]: Loading source files from: {source_files_dir}")
        index_file = source_files_dir / "file_index.json"
        loaded_count = 0
        
        # Load index if it exists
        if index_file.exists():
            try:
                import json
                with open(index_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                total_in_index = len(data)
                skipped_duplicates = 0
                for original_path, (saved_path, file_hash) in data.items():
                    if original_path not in self.file_map:
                        # Extract just the filename from saved_path (might be "source_files/filename.py" or just "filename.py")
                        filename = Path(saved_path).name
                        saved_full_path = source_files_dir / filename
                        
                        if saved_full_path.exists():
                            # Always copy to our source_dir (self.output_base/source_files/)
                            # This ensures files are in a known location relative to self.output_base
                            our_saved_path = self.source_dir / filename
                            if not our_saved_path.exists():
                                try:
                                    import shutil
                                    logging.debug(f"[OpGuard]: Copying source file: {saved_full_path} -> {our_saved_path}")
                                    shutil.copy2(saved_full_path, our_saved_path)
                                except Exception as e:
                                    logging.warning(f"[OpGuard]: Failed to copy source file {saved_full_path} to {our_saved_path}: {e}")
                                    continue
                            
                            # Store relative path from output_base
                            saved_path_rel = f"source_files/{filename}"
                            
                            # Pre-load file content into cache for faster access
                            try:
                                content = _safe_read_file(our_saved_path)
                                if content:
                                    self.file_contents[original_path] = content
                                    logging.debug(f"[OpGuard]: Pre-loaded file content: {original_path} -> {our_saved_path}")
                            except Exception as e:
                                logging.debug(f"[OpGuard]: Failed to pre-load file content for {original_path}: {e}")
                            
                            self.file_map[original_path] = (saved_path_rel, file_hash)
                            self.processed.add(original_path)
                            loaded_count += 1
                            logging.debug(f"[OpGuard]: Loaded source file mapping: {original_path} -> {saved_path_rel}")
                    else:
                        skipped_duplicates += 1
                
                if skipped_duplicates > 0:
                    logging.debug(f"[OpGuard]: Skipped {skipped_duplicates} duplicate files (already loaded from another run)")
            except Exception as e:
                logging.warning(f"[OpGuard]: Failed to load source file index from {index_file}: {e}")
        
        if loaded_count > 0:
            logging.info(f"[OpGuard]: Loaded {loaded_count} source files from {source_files_dir}")
            if skipped_duplicates > 0:
                logging.debug(f"[OpGuard]: Skipped {skipped_duplicates} duplicate files (already loaded)")
        elif index_file.exists() and 'total_in_index' in locals():
            # Index exists but no files were loaded (all duplicates)
            logging.debug(f"[OpGuard]: Index file found with {total_in_index} entries, but all files were already loaded (duplicates)")
        return loaded_count


# Global collector instance (lazy initialization)
_collector: Optional[SourceFileCollector] = None
_collector_lock = None


def load_source_files_from_runs(collector: SourceFileCollector, run1_dir: str, run2_dir: str) -> int:
    """
    Load existing source files from run directories.
    Supports two patterns:
    1. run_dir (trial directory) -> look for run_dir/source_files/
    2. run_dir/gpu_XXX (GPU-specific directory) -> look for run_dir/gpu_XXX/source_files/
    
    Note: source_files are stored inside the gpu_XXX directory when comparing GPU-specific directories.
    
    Args:
        collector: SourceFileCollector instance to load files into
        run1_dir: Path to run1 directory (can be trial directory or gpu_XXX subdirectory)
        run2_dir: Path to run2 directory (can be trial directory or gpu_XXX subdirectory)
        
    Returns:
        Total number of files loaded
    """
    from pathlib import Path
    total_loaded = 0
    
    for run_name, run_dir in [("run1", run1_dir), ("run2", run2_dir)]:
        if not run_dir:
            logging.debug(f"[OpGuard]: Skipping {run_name} (no directory provided)")
            continue
        
        run_path = Path(run_dir)
        if not run_path.exists():
            logging.warning(f"[OpGuard]: {run_name} directory does not exist: {run_path}")
            continue
        
        logging.info(f"[OpGuard]: Processing {run_name} directory: {run_path}")
        
        # Recognize pattern: run_dir/gpu_XXX
        # In this case, source_files are stored inside the gpu_XXX directory
        if run_path.name.startswith("gpu_"):
            # This is a GPU-specific directory, source_files are inside it
            source_files_dir = run_path / "source_files"
            print(f"📂 [{run_name}] Detected gpu_XXX pattern")
            print(f"   Input path: {run_path}")
            print(f"   Looking for source_files in: {source_files_dir}")
            if not source_files_dir.exists():
                print(f"   ⚠️  source_files directory does not exist: {source_files_dir}")
            elif not (source_files_dir / "file_index.json").exists():
                print(f"   ⚠️  file_index.json not found in: {source_files_dir}")
            else:
                print(f"   ✅ Found source_files directory and index file")
            loaded = collector.load_from_directory(run_path)
            total_loaded += loaded
            if loaded > 0:
                print(f"   ✅ Loaded {loaded} new files from {run_name} gpu directory")
            else:
                print(f"   ℹ️  No new files loaded from {run_name} (all files already loaded from previous runs or directory empty)")
        else:
            # This is the trial directory, source_files are here
            source_files_dir = run_path / "source_files"
            print(f"📂 [{run_name}] Looking for source_files in trial directory")
            print(f"   Trial directory: {run_path}")
            print(f"   Expected source_files directory: {source_files_dir}")
            if not source_files_dir.exists():
                print(f"   ⚠️  source_files directory does not exist: {source_files_dir}")
            elif not (source_files_dir / "file_index.json").exists():
                print(f"   ⚠️  file_index.json not found in: {source_files_dir}")
            else:
                print(f"   ✅ Found source_files directory and index file")
            loaded = collector.load_from_directory(run_path)
            total_loaded += loaded
            if loaded > 0:
                print(f"   ✅ Loaded {loaded} new files from {run_name} directory")
            else:
                print(f"   ℹ️  No new files loaded from {run_name} (all files already loaded from previous runs or directory empty)")
    
    logging.info(f"[OpGuard]: Total source files loaded: {total_loaded}")
    return total_loaded

def get_collector(output_base: Optional[str] = None) -> Optional[SourceFileCollector]:
    """
    Get or create the global source file collector.
    
    Args:
        output_base: Base directory for trace output (required on first call)
        
    Returns:
        SourceFileCollector instance, or None if output_base not provided
    """
    global _collector, _collector_lock
    
    if _collector_lock is None:
        import threading
        _collector_lock = threading.Lock()
    
    if output_base:
        output_base_path = Path(output_base).resolve()
        
        # Check if collector exists and if output_base has changed
        if _collector is None:
            with _collector_lock:
                if _collector is None:
                    _collector = SourceFileCollector(output_base)
        else:
            current_base = _collector.output_base.resolve()
            if current_base != output_base_path:
                # Output base has changed, reinitialize the collector
                with _collector_lock:
                    # Double-check after acquiring lock
                    if _collector.output_base.resolve() != output_base_path:
                        # Finalize old collector before creating new one
                        try:
                            _collector.finalize()
                        except Exception as e:
                            logging.warning(f"[OpGuard]: Failed to finalize old collector: {e}")
                        _collector = SourceFileCollector(output_base)
    
    return _collector


def process_trace_stack(trace: dict, output_base: Optional[str] = None) -> dict:
    """
    Process a trace dictionary, extract and save source files from its stack trace.
    
    Args:
        trace: Trace dictionary (may contain "stack_trace" field)
        output_base: Base directory for trace output
        
    Returns:
        Updated trace dictionary with "source_files" mapping added
    """
    has_stack_trace = "stack_trace" in trace
    stack_trace = trace.get("stack_trace")
    
    collector = get_collector(output_base)
    if not collector:
        return trace
    
    if stack_trace:
        try:
            file_mapping = collector.process_stack_trace(stack_trace)
            if file_mapping:
                trace["source_files"] = file_mapping
        except Exception as e:
            logging.error(f"[OpGuard]: Error processing stack trace: {e}", exc_info=True)
    
    return trace

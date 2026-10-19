import threading
from opguard.local_sync.paths import atomic_output
import queue
import os
import sys
import json
import torch
import signal
import re
import atexit
from collections import defaultdict
from opguard._running import get_output_subdir
from opguard.utils.utils import get_id, register_signal_handler
from opguard.tracer.source_files import get_collector, process_trace_stack
import numpy as np
import logging


TRACE_BATCH_MAX_COUNT = 10000
TENSOR_BATCH_MAX_COUNT = 1000
TRACE_BATCH = 1000
TENSOR_BATCH = 100
WORKER_TIMEOUT = 300


def _sanitize_function_name(func_name: str) -> str:
    """
    Sanitize function name for use in filesystem paths.
    Replaces invalid characters with underscores and limits length.
    """
    if not func_name:
        return "unknown"
    # Remove angle brackets and replace invalid chars
    sanitized = func_name.replace("<", "").replace(">", "")
    # Replace invalid filesystem characters with underscores
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', sanitized)
    # Replace multiple dots/underscores with single underscore
    sanitized = re.sub(r'[._]{2,}', '_', sanitized)
    # Remove leading/trailing dots and underscores
    sanitized = sanitized.strip('._')
    # Limit length to avoid filesystem issues
    if len(sanitized) > 200:
        # Use hash for very long names
        import hashlib
        hash_suffix = hashlib.md5(func_name.encode()).hexdigest()[:8]
        sanitized = sanitized[:190] + '_' + hash_suffix
    # Ensure it's not empty
    if not sanitized:
        sanitized = "unknown"
    return sanitized


def _json_default(obj):
    """
    Best-effort JSON serializer for trace payloads.

    Goal: never drop an entire trace record just because one meta-var is a custom
    Python object (e.g., framework-specific classes). We first try structured
    conversions for common tensor/numpy/container types, then fall back to str().
    """
    try:
        # Torch tensors: keep them compact; trace JSONL should not contain large tensors
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist() if obj.numel() <= 1024 else {
                "__opguard_tensor__": True,
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "device": str(obj.device),
                "numel": int(obj.numel()),
            }
        # Numpy scalars / arrays
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # Bytes
        if isinstance(obj, (bytes, bytearray)):
            return obj.decode("utf-8", errors="replace")
        # Sets
        if isinstance(obj, set):
            return list(obj)
    except Exception:
        # If any of the structured conversions fail, fall through to str()
        pass

    # Last resort: stringify (this is what fixes "Object of type X is not JSON serializable")
    try:
        return str(obj)
    except Exception:
        # Extremely defensive: if even str() fails, drop the value but keep the trace
        return None

# --- Singleton lazy init for TraceBuffer ---
TRACE_BUFFER = None
LOCK = threading.Lock()

# Fork-safety: reset singleton + lock in child, and mark any inherited buffer as stale.
def _after_fork_child():
    global TRACE_BUFFER, LOCK
    try:
        if TRACE_BUFFER is not None:
            # Mark inherited buffer as stale so stop() becomes a no-op in child.
            TRACE_BUFFER._forked = True
    except Exception:
        pass
    TRACE_BUFFER = None
    LOCK = threading.Lock()

if hasattr(os, "register_at_fork"):
    try:
        os.register_at_fork(after_in_child=_after_fork_child)
    except Exception:
        # Best-effort: if registration fails, we'll fall back to PID checks.
        pass


class TraceBuffer:
    def __init__(self):
        self.id = get_id()
        self._pid = os.getpid()
        self._forked = False
        self.buffer_lock = threading.Lock()
        self.trace_buffer = defaultdict(list)    # func -> list[trace]
        self.trace_count = 0
        self.tensor_buffer = []
        self.file_handles = {}
        self.file_handles_lock = threading.Lock()  # Protect file_handles access
        self.dump_queue = queue.Queue()
        self.stopping = False
        self.closed = False
        self.stop_lock = threading.Lock()  # Protect stop() from concurrent calls
        self._in_signal_handler = False  # Track if we're in signal handler context

        self.worker_thread = threading.Thread(target=self.dump_worker, daemon=True)
        self.worker_thread.start()
        
        # Ensure final dump is drained on interpreter exit.
        atexit.register(self._stop_on_exit)
        self.install_chained_signal_handlers()

        logging.info(f"[OpGuard]: trace buffer {self.id} init")

    def add_trace(self, tag, trace):
        """producer: add trace to buffer"""
        if self.stopping or self.closed:
            return
        with self.buffer_lock:
            self.trace_buffer[tag].append(trace)
            self.trace_count += 1

            if self.trace_count >= TRACE_BATCH_MAX_COUNT:
                # emegrency dump buffer to queue
                batch = []
                for traces in self.trace_buffer.values():
                    batch.extend(traces)

                self.trace_buffer.clear()
                self.trace_count = 0
                if not self.stopping and not self.closed:
                    self.dump_queue.put(("summary", batch))
                    

    def flush(self):
        if self.stopping or self.closed:
            return            
        with self.buffer_lock:
            if self.trace_count >= TRACE_BATCH:
                # transform buffer to queue to dump
                trace_batch = []
                for traces in self.trace_buffer.values():
                    trace_batch.extend(traces)

                self.trace_buffer.clear()
                self.trace_count = 0
                if not self.stopping and not self.closed:
                    self.dump_queue.put(("summary", trace_batch))
            
            if len(self.tensor_buffer) >= TENSOR_BATCH:
                tensor_batch = self.tensor_buffer
                self.tensor_buffer = []
                if not self.stopping and not self.closed:
                    self.dump_queue.put(("full", tensor_batch))

    
    def add_full_tensors(self, tensor):
        """producer: add full tensor to buffer"""
        if self.stopping or self.closed:
            return
        with self.buffer_lock:
            self.tensor_buffer.append(tensor)
            
            # emegrency dump buffer to queue
            if len(self.tensor_buffer) >= TENSOR_BATCH_MAX_COUNT:
                tensor_batch = self.tensor_buffer
                self.tensor_buffer = []
                if not self.stopping and not self.closed:
                    self.dump_queue.put(("full", tensor_batch))
    
    def dump_worker(self):
        """consumer: dump trace and full tensor from queue"""
        
        logging.info(f"[OpGuard]: trace buffer {self.id} start dump worker")
        
        while True:
            try:
                kind, data = self.dump_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            
            try:
                if kind == "stop":
                    logging.info(f"[OpGuard]: trace buffer {self.id} dump worker receive stop signal and exit")
                    break
                if kind == "summary":
                    logging.info(f"[OpGuard]: trace buffer {self.id} dump worker receive summary batch with {len(data)} traces, remaining items: {self.dump_queue.unfinished_tasks-1}")
                    self.finalize_and_dump_traces(data)
                elif kind == "full":
                    logging.info(f"[OpGuard]: trace buffer {self.id} dump worker receive full tensor batch with {len(data)} tensors, remaining items: {self.dump_queue.unfinished_tasks-1}")
                    self.finalize_and_dump_tensors(data)
            except Exception as e:
                logging.error(f"[OpGuard]: trace buffer {self.id}] dump worker failed to process {kind} item: {e}")
            finally:
                self.dump_queue.task_done()

    def finalize_and_dump_traces(self, traces):
        for trace in traces:
            trace["inputs"] = self.finalize_io(trace.get("inputs", {}))
            trace["outputs"] = self.finalize_io(trace.get("outputs", {}))
            self.write_trace(trace)

    def finalize_and_dump_tensors(self, traces):
        for trace in traces:
            self.write_tensor(trace)

    def _resolve_trace_output_base(self, tag_prefix: str):
        """JSONL and tensor dumps always share the same execution directory."""
        original = os.getenv("OPGUARD_EXEC_OUTPUT_DIR") or os.getenv("OPGUARD_TRACE_OUTPUT_BASE")
        return get_output_subdir("exec_trace"), original

    def _resolve_tensor_output_base(self, tag_prefix: str) -> str:
        return get_output_subdir("exec_trace")

    def _build_trace_path(self, trace, output_base: str):
        line_num = trace.get("line_num", "unknown")
        func_name = trace["function"]
        sanitized_func_name = _sanitize_function_name(func_name)
        key = (trace["gpu_rank"], func_name, trace["tag_prefix"], line_num)
        filename = f"{sanitized_func_name}_line{line_num}"
        if trace.get("is_async", False):
            filename += "_async"
        filename += ".jsonl"
        path = os.path.join(
            output_base,
            f"gpu_{trace['gpu_rank']:03d}",
            "ops",
            sanitized_func_name,
            filename,
        )
        return key, path

    def _get_or_create_trace_file(self, key, path):
        with self.file_handles_lock:
            fh = self.file_handles.get(key)
            if fh is not None:
                return fh

            os.makedirs(os.path.dirname(path), exist_ok=True)
            is_shadow = "/tmp/opguard_local_dump" in path or "opguard_local_dump" in path
            if is_shadow:
                logging.info(f"[OpGuard]: Creating trace file in local shadow directory: {path}")
            try:
                fh = open(path, "a", buffering=1)
                self.file_handles[key] = fh
                return fh
            except Exception as e:
                logging.error(f"[OpGuard]: trace buffer {self.id}] create file {path} failed: {e}")
                return None

    @staticmethod
    def _write_jsonl_line(fh, trace):
        json.dump(trace, fh, default=_json_default, ensure_ascii=False)
        fh.write("\n")

    def _write_trace_line(self, fh, trace):
        """Write one JSONL record using simple append semantics."""
        self._write_jsonl_line(fh, trace)

    def write_trace(self, trace):
        output_base, original_output_base = self._resolve_trace_output_base(trace["tag_prefix"])

        if original_output_base and output_base != original_output_base:
            logging.info(f"[OpGuard]: Runtime trace writing to local shadow directory: {output_base} (target: {original_output_base})")

        if trace.get("stack_trace"):
            try:
                process_trace_stack(trace, output_base)
            except Exception as e:
                # Don't fail trace writing if source file collection fails
                logging.error(f"[OpGuard]: Failed to collect source files for trace: {e}", exc_info=True)

        key, path = self._build_trace_path(trace, output_base)
        fh = self._get_or_create_trace_file(key, path)
        if fh is None:
            return

        try:
            self._write_trace_line(fh, trace)
        except Exception as e:
            logging.error(f"[OpGuard]: trace buffer {self.id}] write trace to file {fh.name} failed: {e}")

    def write_tensor(self, trace):
        base = self._resolve_tensor_output_base(trace["tag_prefix"])
        
        meta = trace.get("meta", {})
        gs = meta.get("global_step", "NA")
        mbidx = meta.get("microbatch_idx", meta.get("micro_batch_idx", "NA"))
        layer = meta.get("layer_number", meta.get("layer_num", "NA"))
        seq_dir = f"gs{gs}_mb{mbidx}_layer{layer}"

        # Write tensor files to ops/full/ subdirectory within gpu_XXX
        subdir = os.path.join(
            base,
            f"gpu_{trace['gpu_rank']:03d}",
            "ops",
            "full",
            trace["function"], seq_dir
        )
        os.makedirs(subdir, exist_ok=True)

        def dump(obj, prefix):
            if isinstance(obj, torch.Tensor):
                safe = prefix.replace("[", "_").replace("]", "").replace(".", "_")
                fname = f"{trace['fname']}_line{trace['lineno']}_{safe}.pt"
                with atomic_output(os.path.join(subdir, fname)) as temporary:
                    torch.save(obj, temporary)
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    dump(v, f"{prefix}.{k}")
            elif isinstance(obj, (list, tuple)):
                for i, v in enumerate(obj):
                    dump(v, f"{prefix}[{i}]")

        dump(trace.get("inputs", {}), "input")
        dump(trace.get("outputs", {}), "output")

    def finalize_io(self, obj):
        if isinstance(obj, dict):
            return {k: self.finalize_io(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.finalize_io(v) for v in obj]
        elif isinstance(obj, torch.Tensor):
            return float(obj.cpu().item()) if obj.numel() == 1 else obj.cpu().tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def force_dump_all(self):
        """force dump all traces and tensors"""
        print(f"[OpGuard]: trace buffer {self.id} force dump all traces and tensors", file=sys.stderr, flush=False)
        with self.buffer_lock:
            if self.trace_buffer:
                traces = [trace for lst in self.trace_buffer.values() for trace in lst]
                self.trace_buffer.clear()
                self.trace_count = 0
            else:
                traces = []

            if self.tensor_buffer:
                tensors = self.tensor_buffer
                self.tensor_buffer = []
            else:
                tensors = []

        if traces:
            self.dump_queue.put(("summary", traces))
        if tensors:
            self.dump_queue.put(("full", tensors))


    def close_all_files(self):
        # Acquire lock to safely access and close file handles
        with self.file_handles_lock:
            handles_to_close = list(self.file_handles.values())
            self.file_handles.clear()
        
        # Close handles outside the lock to avoid blocking
        for f in handles_to_close:
            try:
                f.close()
            except Exception as e:
                # Use safe logging to avoid reentrant issues during signal handling
                try:
                    logging.error(f"[OpGuard]: trace buffer {self.id} close file handle failed: {e}")
                except RuntimeError:
                    try:
                        print(f"[OpGuard]: trace buffer {self.id} close file handle failed: {e}", file=sys.stderr, flush=False)
                    except Exception:
                        pass
        try:
            logging.info(f"[OpGuard]: trace buffer {self.id} close all file handles")
        except RuntimeError:
            try:
                print(f"[OpGuard]: trace buffer {self.id} close all file handles", file=sys.stderr, flush=False)
            except Exception:
                pass

    def install_chained_signal_handlers(self):
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                original = signal.getsignal(sig)

                def make_chained_handler(sig, original):
                    def handler(signum, frame):
                        # Mark that we're in a signal handler to avoid logging
                        self._in_signal_handler = True
                        # Write directly to stderr to avoid reentrant logging issues during signal handling
                        try:
                            print(f"[OpGuard]: trace buffer {self.id} received signal {signum}", file=sys.stderr, flush=False)
                        except Exception:
                            pass  # Ignore errors during signal handling

                        self._stop_on_exit()

                        if callable(original):
                            original(signum, frame)
                        elif original == signal.SIG_DFL:
                            signal.signal(sig, signal.SIG_DFL)
                            if hasattr(signal, 'raise_signal'):
                                signal.raise_signal(sig)
                            else:
                                # Python < 3.8 fallback
                                os.kill(os.getpid(), sig)

                    return handler

                signal.signal(sig, make_chained_handler(sig, original))

            except Exception as e:
                logging.warning(f"[OpGuard]: install signal handler for {sig} failed: {e}")

    
    def stop(self):
        # If this buffer was created in a different process (forked child),
        # do not attempt to join the worker thread or touch locks.
        if self._pid != os.getpid() or getattr(self, "_forked", False):
            try:
                self.stopping = True
                self.closed = True
            except Exception:
                pass
            return
        # Use lock to prevent concurrent calls to stop() from causing deadlock
        with self.stop_lock:
            if self.stopping or self.closed:
                return
            self.stopping = True
        
        # Track if we've encountered a reentrant error in this stop() call
        # Once we hit one, prefer stderr for all subsequent logs
        _use_stderr_only = [self._in_signal_handler]  # Use list for mutable closure

        # Use safe logging that won't cause reentrant issues during signal handling
        def safe_log(level, msg):
            """Safely log a message, catching reentrant errors during signal handling.
            
            During signal handling or multiprocessing spawn, stdout/stderr may be in a
            reentrant flush state. We detect this and skip logging entirely, writing
            directly to stderr instead.
            """
            # If we're in a signal handler or have already hit a reentrant error, skip logging entirely
            if _use_stderr_only[0] or self._in_signal_handler:
                return
            
            # Try logging, but catch any reentrant errors
            try:
                if level == 'info':
                    logging.info(msg)
                elif level == 'warning':
                    logging.warning(msg)
                elif level == 'error':
                    logging.error(msg)
            except (RuntimeError, OSError, ValueError) as e:
                # Reentrant call or other I/O error during signal handling - write directly to stderr instead
                # This can happen during multiprocessing spawn when stdout is being flushed
                # Once we hit this, prefer stderr for all subsequent logs in this stop() call
                _use_stderr_only[0] = True
                try:
                    print(msg, file=sys.stderr, flush=False)
                except Exception:
                    pass  # Ignore errors during signal handling

        safe_log('info', f"[OpGuard]: trace buffer {self.id} receive stop signal and dump trace buffer")
        self.force_dump_all()

        self.dump_queue.put(("stop", None))

        safe_log('info', f"[OpGuard]: trace buffer {self.id} wait for dump queue empty")
        try:
            # Block until all queued items are processed, including the stop sentinel.
            self.dump_queue.join()
            # Worker should exit after processing stop sentinel.
            self.worker_thread.join()
        except Exception as e:
            safe_log('error', f"[OpGuard]: trace buffer {self.id} dump worker join exception: {e}")

        # Only close files after worker thread has finished (or timed out)
        # This prevents closing files while worker thread is still using them
        self.close_all_files()
        
        # Finalize source file collector to save index
        try:
            from opguard.tracer.source_files import get_collector
            # Get output base from any trace if available, or use default
            collector = get_collector()
            if collector:
                collector.finalize()
                safe_log('info', f"[OpGuard]: Finalized source file collection")
        except Exception as e:
            safe_log('info', f"[OpGuard]: Failed to finalize source file collector: {e}")

        with self.stop_lock:
            self.closed = True
            # Reset signal handler flag after stop completes
            self._in_signal_handler = False

    def _stop_on_exit(self):
        """Drain all buffered traces on interpreter shutdown."""
        try:
            self.stop()
        except Exception:
            # Never let atexit fail the process exit path.
            pass

def get_trace_buffer() -> TraceBuffer:
    global TRACE_BUFFER
    global LOCK
    # If we forked after creating the buffer, drop the inherited instance.
    if TRACE_BUFFER is not None and getattr(TRACE_BUFFER, "_pid", None) != os.getpid():
        TRACE_BUFFER = None
        LOCK = threading.Lock()
    if TRACE_BUFFER is None:
        with LOCK:
            if TRACE_BUFFER is None:  # double-checked locking
                TRACE_BUFFER = TraceBuffer()
    return TRACE_BUFFER

def ADD_TRACE(tag, trace):
    buf = get_trace_buffer()
    return buf.add_trace(tag, trace)

def ADD_FULL_TENSORS(tensor):
    buf = get_trace_buffer()
    return buf.add_full_tensors(tensor)

def FLUSH():
    buf = get_trace_buffer()
    return buf.flush()

def FORCE_DUMP_ALL():
    buf = get_trace_buffer()
    return buf.force_dump_all()

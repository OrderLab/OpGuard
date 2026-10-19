#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import signal
import atexit
import sys

def main():
    # Configuration
    num_handlers  = 100
    total_records = 10_000      # total writes attempted
    kill_after    = 5_000       # record index at which to kill the program
    output_dir    = "./test_jsonl_file_handler"

    # Prepare output directory
    os.makedirs(output_dir, exist_ok=True)

    # Open 100 file handlers
    handlers = []
    for i in range(num_handlers):
        path = os.path.join(output_dir, f"test_{i:03d}.jsonl")
        f = open(path, "w", encoding="utf-8")
        handlers.append(f)

    # Register cleanup to close all handlers on normal exit
    def _close_handlers():
        for fh in handlers:
            try:
                fh.close()
            except:
                pass
    atexit.register(_close_handlers)

    # Handle softer termination signals so that atexit can run
    def _on_exit(signum, frame):
        sys.exit(0)
    signal.signal(signal.SIGTERM, _on_exit)
    signal.signal(signal.SIGINT,  _on_exit)

    for i in range(total_records):
        # Select handler in round-robin
        idx = i % num_handlers
        f   = handlers[idx]

        # Build a simple test record
        record = {
            "timestamp": time.time(),
            "global_id": i,
            "file_idx": idx,
            "message": f"Record {i} to file {idx}"
        }

        # Write, flush immediately
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        time.sleep(0.01)

if __name__ == "__main__":
    main()

# Callsite Tracing with Python + CUDA Hooks

This tracer captures Python callstacks for CUDA/cuBLAS/cuBLASLt operations and writes them to JSON files. It supports **live streaming** (for real-time inspection) and/or **aggregate summaries** (at program exit).

---

## 1. Build & Load

1. Compile the shared object (`.so`) and preload it:

   ```bash
   g++ -fPIC -shared -std=c++17 -o libpyhook.so hook.cpp -ldl -pthread -I${PYTHON_INCLUDE}
   export LD_PRELOAD=/path/to/libpyhook.so
   ```
2. Run your Python program as usual:

   ```bash
   python train.py
   ```

---

## 2. Configure Output

Set environment variables before running:

* **Output file template**

  ```bash
  export PYHOOK_JSON_TPL="/logs/pyhook_{rank}_{pid}_{tid}.ndjson"
  ```

  Placeholders:

  * `{pid}` process id
  * `{tid}` thread id
  * `{rank}` MPI/SLURM rank (auto-detected)
  * `{hostname}` machine name
  * `{time}` timestamp

* **Streaming mode** (write events immediately):

  ```bash
  export PYHOOK_STREAM_JSON=1
  ```

* **Deduplicate stacks during streaming (optional):**

  ```bash
  export PYHOOK_STREAM_DEDUP=1
  ```

* **Aggregate summary at exit:**
  always written to `*.aggregate.json` next to the NDJSON file.
  Suppressed if no events were seen.

---

## 3. Inspect Logs

* **While running:**

  ```bash
  tail -f /logs/pyhook_0_12345_6789.ndjson
  ```

  Each line is a JSON object with:

  ```json
  {
    "site": "train.py:42 in forward",
    "op": "cudaLaunchKernel",
    "frames": [["train.py", 42, "forward"], ...]
  }
  ```

* **After exit:**
  Open the aggregate summary:

  ```bash
  less /logs/pyhook_0_12345.aggregate.json
  ```

  Contains per-site counts, unique callstacks, and ops.

---

## 4. Tips

* **Per-process vs per-thread:**

  * Use `{pid}` for one file per process.
  * Add `{tid}` to also split by thread (avoids locking).

* **No empty files:**
  If a process/thread saw no CUDA ops, no file is written.

* **Real-time + summary:**
  You can enable both streaming (live NDJSON) and aggregate (exit JSON).

---

👉 With this setup, you can watch CUDA API usage in real time **and** get a compact summary for offline analysis.

---
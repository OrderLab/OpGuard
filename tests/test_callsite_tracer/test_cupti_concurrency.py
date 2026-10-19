"""Compile the real bridge against a small CUPTI double; no GPU is required.

The double models a native callback thread which must finish while its caller
holds the GIL. A subprocess timeout makes a lock inversion a bounded failure.
This tests host-side locking, not CUDA ABI or driver behavior.
"""
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[2] / "opguard/callsite_tracer/cupti_tracer/opguard_cupti.cpp"

CUDA = r"""
#pragma once
#include <cstdint>
using CUcontext = void*;
using CUstream = void*;
using CUgraph = void*;
using CUgraphExec = void*;
// Intentionally no cuCtxGetDevice/cuCtxGetCurrent declarations: callbacks
// must use CUPTI's callback-safe queries, not reenter the CUDA driver.
struct cuStreamBeginCapture_params { CUstream hStream; };
struct cuStreamEndCapture_params { CUstream hStream; CUgraph* phGraph; };
struct cuGraphInstantiate_params { CUgraphExec* phGraphExec; CUgraph hGraph; };
struct cuLaunchKernel_params { CUstream hStream; };
struct cudaLaunchKernel_v7000_params { CUstream stream; };
"""
CUPTI = r"""
#pragma once
#include "cuda.h"
#include <cstddef>
#define CUPTIAPI
#define CUPTI_API_VERSION 13
using CUptiResult = int;
using CUpti_CallbackDomain = int;
using CUpti_CallbackId = int;
using CUpti_SubscriberHandle = void*;
using CUpti_CallbackFunc = void(*)();
enum { CUPTI_SUCCESS = 0, CUPTI_API_ENTER = 1, CUPTI_API_EXIT,
 CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_CB_DOMAIN_RUNTIME_API,
 CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCapture,
 CUPTI_DRIVER_TRACE_CBID_cuStreamEndCapture,
 CUPTI_DRIVER_TRACE_CBID_cuGraphInstantiate,
 CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel,
 CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000,
 CUPTI_ACTIVITY_KIND_KERNEL, CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL };
struct CUpti_CallbackData {
 int callbackSite; const void* functionParams; uint32_t correlationId;
 const char* functionName; CUcontext context;
};
struct CUpti_Activity { int kind; };
struct CUpti_ActivityKernel8 {
 int kind; const char* name; uint32_t correlationId, deviceId, streamId;
 uint64_t start, end;
};
inline int cuptiGetStreamId(CUcontext, CUstream, uint32_t* id) { *id=11; return 0; }
inline int cuptiGetDeviceId(CUcontext, uint32_t* id) { *id=0; return 0; }
inline int cuptiGetResultString(int, const char** s) { *s="test"; return 0; }
inline int cuptiSubscribe(CUpti_SubscriberHandle* s, CUpti_CallbackFunc, void*) { *s=(void*)1; return 0; }
inline int cuptiEnableCallback(int, void*, int, int) { return 0; }
inline int cuptiEnableDomain(int, void*, int) { return 0; }
inline int cuptiActivityEnable(int) { return 0; }
inline int cuptiActivityRegisterCallbacks(void(*)(uint8_t**, size_t*, size_t*),
 void(*)(CUcontext,uint32_t,uint8_t*,size_t,size_t)) { return 0; }
inline int cuptiActivityFlushAll(int) { return 0; }
inline int cuptiUnsubscribe(void*) { return 0; }
inline int cuptiActivityGetNextRecord(uint8_t*, size_t, CUpti_Activity**) { return 1; }
inline int cuptiActivityGetNumDroppedRecords(CUcontext, uint32_t, size_t* n) { *n=0; return 0; }
"""
WRAPPER = r"""
#include <thread>
#define PyInit_opguard_cupti bridge_init
#include "BRIDGE_SOURCE"
#undef PyInit_opguard_cupti
static void launch() {
 cudaLaunchKernel_v7000_params p{(CUstream)1};
 CUpti_CallbackData info{CUPTI_API_ENTER, &p, 123, "cudaLaunchKernel", (CUcontext)1};
 api_cb(nullptr, CUPTI_CB_DOMAIN_RUNTIME_API, CUDA_LAUNCH_CBID, &info);
 CUpti_ActivityKernel8 k{CUPTI_ACTIVITY_KIND_KERNEL, "test_kernel", 123, 0, 11, 1, 2};
 std::vector<Frame> stack;
 { std::lock_guard<std::mutex> lk(g_mutex); stack = g_corr2stack_runtime.at(123); }
 enqueue_event_mem(&k, &stack);
}
static PyObject* worker(PyObject*, PyObject*) {
 // Model a Python thread waiting on native work while still holding the GIL.
 // The old PyGILState_Ensure in api_cb deadlocks here deterministically.
 std::thread t(launch);
 t.join();
 Py_RETURN_NONE;
}
static PyObject* detached(PyObject*, PyObject*) {
 Py_BEGIN_ALLOW_THREADS
 launch();
 Py_END_ALLOW_THREADS
 Py_RETURN_NONE;
}
static PyObject* held(PyObject*, PyObject*) { launch(); Py_RETURN_NONE; }
static PyObject* graph_phases(PyObject*, PyObject*) {
 // ENTER must not dereference the not-yet-written EndCapture output pointer.
 cuStreamEndCapture_params dp{(CUstream)1, (CUgraph*)1};
 CUpti_CallbackData d{CUPTI_API_ENTER, &dp, 1, "cuStreamEndCapture", (CUcontext)1};
 api_cb(nullptr, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuStreamEndCapture, &d);
 RT_cudaStreamEndCapture_params rp{(void*)1, (void**)1};
 d.functionParams = &rp; d.functionName = "cudaStreamEndCapture";
 api_cb(nullptr, CUPTI_CB_DOMAIN_RUNTIME_API, 0, &d);
 RT_cudaGraphLaunch_params lp{(void*)1, nullptr};
 d.functionParams = &lp; d.functionName = "cudaGraphLaunch";
 api_cb(nullptr, CUPTI_CB_DOMAIN_RUNTIME_API, 0, &d);
 StreamIdKey key{0, 11}; // The double assigns 11 even for the default stream.
 if (g_replay_by_sid.find(key) == g_replay_by_sid.end())
   return PyErr_Format(PyExc_AssertionError, "default stream ID was guessed");
 g_replay_by_sid[key].next_idx = 3;
 d.callbackSite = CUPTI_API_EXIT;
 api_cb(nullptr, CUPTI_CB_DOMAIN_RUNTIME_API, 0, &d);
 if (g_replay_by_sid[key].next_idx != 3)
   return PyErr_Format(PyExc_AssertionError, "graph EXIT reset replay progress");
 Py_RETURN_NONE;
}
static PyMethodDef extra[] = {
 {"graph_phases", graph_phases, METH_NOARGS, nullptr},
 {"worker", worker, METH_NOARGS, nullptr},
 {"detached", detached, METH_NOARGS, nullptr},
 {"held", held, METH_NOARGS, nullptr},
 {nullptr, nullptr, 0, nullptr}
};
PyMODINIT_FUNC PyInit_opguard_cupti() {
 PyObject* m = bridge_init();
 if (m && PyModule_AddFunctions(m, extra) < 0) { Py_DECREF(m); return nullptr; }
 return m;
}
"""


class CuptiConcurrencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shlex.split(os.environ.get("CXX", "c++"))
        if not shutil.which(compiler[0]):
            raise unittest.SkipTest("C++ compiler required")
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        root = Path(cls.tmp.name)
        (root / "cuda.h").write_text(CUDA)
        (root / "cupti.h").write_text(CUPTI)
        (root / "wrapper.cpp").write_text(WRAPPER.replace("BRIDGE_SOURCE", SOURCE.as_posix()))
        output = root / ("opguard_cupti" + sysconfig.get_config_var("EXT_SUFFIX"))
        cmd = compiler + ["-std=c++17", "-shared", "-fPIC", "-pthread",
                          "-I" + str(root), "-I" + sysconfig.get_path("include"),
                          str(root / "wrapper.cpp"), "-o", str(output)]
        if sys.platform == "darwin":
            cmd += ["-undefined", "dynamic_lookup"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode:
            raise RuntimeError(result.stderr)

    def run_case(self, code):
        env = {**os.environ, "PYTHONPATH": self.tmp.name, "OPGUARD_AUTOSTART": "0",
               "OPGUARD_STACKS": "1"}
        subprocess.run([sys.executable, "-c", "import opguard_cupti as c\nc.start_mem()\n" + code],
                       cwd=self.tmp.name, env=env, check=True, timeout=10,
                       capture_output=True, text=True)

    def test_worker_callback_does_not_wait_for_gil(self):
        self.run_case("c.worker()\ne = c.drain()\nassert len(e) == 1 and e[0]['stack'] == []\nc.stop()")

    def test_detached_thread_uses_nested_dispatch_snapshot(self):
        self.run_case("""
a = [['outer.py', 7, 'outer']]
b = [['inner.py', 9, 'inner']]
x = c.tag_stream_push(0, 1, a, 'outer')
y = c.tag_stream_push(0, 1, b, 'inner')
c.detached()
assert c.drain()[0]['stack'] == b
c.tag_stream_pop(0, 1, y)
c.detached()
assert c.drain()[0]['stack'] == a
# Worker must not inherit this thread's dispatch tag.
c.worker()
assert c.drain()[0]['stack'] == []
c.tag_stream_pop(0, 1, x)
c.detached()
assert c.drain()[0]['stack'] == []
c.stop()
""")

    def test_graph_callback_phases_and_default_stream(self):
        self.run_case("c.graph_phases()\nc.stop()")

    def test_held_gil_still_captures_stack(self):
        self.run_case("def caller(): c.held()\ncaller()\nassert c.drain()[0]['stack'][-1][2] == 'caller'\nc.stop()")

    def test_tags_validate_inputs_and_pop_order(self):
        self.run_case("""
for bad in ([['x', 1]], [['x', 'bad', 'f']], [None]):
    try: c.tag_stream_push(0, 1, bad)
    except (TypeError, ValueError): pass
    else: raise AssertionError('invalid stack accepted')
x = c.tag_stream_push(0, 1, [['x', 1, 'f']])
try: c.tag_stream_pop(0, 1, x + 1)
except ValueError: pass
else: raise AssertionError('invalid token accepted')
c.detached()
assert c.drain()[0]['stack'] == [['x', 1, 'f']]
c.tag_stream_pop(0, 1, x)
c.set_py_stack_capture(False)
c.held()
assert c.drain()[0]['stack'] == []
c.stop()
""")


@unittest.skipUnless(os.getenv("OPGUARD_TEST_CUPTI_GPU") == "1", "opt-in NVIDIA GPU test")
class CuptiGpuConcurrencyTests(unittest.TestCase):
    def test_concurrent_launch_flush_and_dispatch_attribution(self):
        # The subprocess bounds a real driver hang, including teardown/atexit.
        code = r"""
import concurrent.futures
import torch
from opguard.callsite_tracer import opguard_cupti as c
assert torch.cuda.is_available()
torch.cuda.init()
c.start_mem()
def work(worker):
    with torch.cuda.stream(torch.cuda.Stream()):
        stream = int(torch.cuda.current_stream().cuda_stream)
        stack = [['cupti_concurrency.py', worker + 1, 'worker']]
        for _ in range(30):
            token = c.tag_stream_push(0, stream, stack, 'test')
            try:
                x = torch.ones(1024, device='cuda')
                y = x.sin() + x
            finally:
                c.tag_stream_pop(0, stream, token)
        torch.cuda.current_stream().synchronize()
try:
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(work, i) for i in range(2)]
        futures.append(pool.submit(lambda: [c.flush() for _ in range(30)]))
        for future in futures:
            future.result()
    torch.cuda.synchronize()
    c.flush()
    events = c.drain(100000)
    assert events
    for line in (1, 2):
        assert any(e['stack'] == [['cupti_concurrency.py', line, 'worker']] for e in events)
finally:
    c.stop()
"""
        subprocess.run([sys.executable, "-c", code], check=True, timeout=120,
                       env={**os.environ, "OPGUARD_AUTOSTART": "0", "OPGUARD_STACKS": "1"})


if __name__ == '__main__':
    unittest.main()

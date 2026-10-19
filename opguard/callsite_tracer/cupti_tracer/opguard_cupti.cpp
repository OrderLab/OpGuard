// opguard_cupti.cpp
// Build:
// g++ -O2 -fPIC -shared opguard_cupti.cpp -o opguard_cupti$(python3-config --extension-suffix) \
//   $(python3-config --includes) -I/usr/local/cuda/include -L/usr/local/cuda/lib64 \
//   -lcupti -lcuda -lpthread -ldl

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <frameobject.h>

#include <cuda.h>
#include <cupti.h>

#include <unistd.h>
#include <inttypes.h>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <deque>
#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>

// ------------------- small helpers -------------------
static inline std::string s_strn(const char* p, size_t cap = (1u<<16)) {
  if (!p) return std::string();
  size_t n = strnlen(p, cap);
  if (n == cap) return std::string();
  return std::string(p, n);
}

static std::string jesc(const std::string& s){
  std::string o; o.reserve(s.size()+8);
  for(char c: s){
    switch(c){
      case '\\': o += "\\\\"; break;
      case '"':  o += "\\\""; break;
      case '\n': o += "\\n";  break;
      case '\r': o += "\\r";  break;
      case '\t': o += "\\t";  break;
      default:   o += c;      break;
    }
  }
  return o;
}

// ------------------- data types -------------------
struct Frame { std::string file; int line; std::string func; };

struct StreamKey { int device; CUstream h; };
struct KeyHash {
  size_t operator()(StreamKey const& k) const noexcept {
    return (size_t)k.device ^ ((size_t)k.h >> 4);
  }
};
struct KeyEq {
  bool operator()(StreamKey const& a, StreamKey const& b) const noexcept {
    return a.device == b.device && a.h == b.h;
  }
};

struct StreamIdKey { int device; uint32_t sid; };
struct SidHash {
  size_t operator()(StreamIdKey const& k) const noexcept {
    return (size_t)k.device ^ ((size_t)k.sid << 1);
  }
};
struct SidEq {
  bool operator()(StreamIdKey const& a, StreamIdKey const& b) const noexcept {
    return a.device == b.device && a.sid == b.sid;
  }
};

struct Event {
  std::string kernel;
  uint32_t correlation;
  uint32_t device;
  uint32_t stream;
  uint64_t start;
  uint64_t end;
  std::vector<Frame> stack; // root->leaf
};

// --- CUDA Graph capture/replay mapping (sequence-based) ----------------------
struct CaptureSession {
  bool active = false;
  std::vector<std::vector<Frame>> node_stacks; // capture order
};

static std::unordered_map<CUstream, CaptureSession> g_capture_by_stream;
static std::unordered_map<CUgraph, std::vector<std::vector<Frame>>> g_graph_to_node_stacks;
static std::unordered_map<CUgraphExec, CUgraph> g_exec_to_graph;

// Sticky mapping: last stack used for this (device, streamId) during graph replay
static std::unordered_map<StreamIdKey, std::vector<Frame>, SidHash, SidEq> g_last_stack_by_sid;


// Per-(device,streamId) replay progress (assign stacks in order during replay)
struct ReplayProgress {
  CUgraphExec exec  = nullptr;  // preferred if available
  CUgraph     graph = nullptr;  // fallback when only hGraph is exposed
  size_t      next_idx = 0;     // next node index to assign
};

static std::unordered_map<StreamIdKey, ReplayProgress, SidHash, SidEq> g_replay_by_sid;

static bool dbg_on() {
  static int on = [](){
    const char* p = getenv("OPGUARD_DEBUG");
    return (p && *p && strcmp(p, "0") != 0);
  }();
  return on;
}
#define DBG(fmt, ...) do { if (dbg_on()) fprintf(stderr, "[opguard_cupti] " fmt "\n", ##__VA_ARGS__); } while(0)

static void dbg_dump_stack(const std::vector<Frame>& st, const char* prefix) {
  if (!dbg_on()) return;
  fprintf(stderr, "[opguard_cupti] %s stack_depth=%zu\n", prefix, st.size());
  // print a few root frames and the leaf so you see line numbers you care about
  size_t n = st.size();
  for (size_t i = 0; i < n; ++i) {
    const auto& fr = st[i];
    // limit spam: first 3 + last 1
    if (i < 3 || i + 1 == n) {
      fprintf(stderr, "  #%zu %s:%d in %s\n", i, fr.file.c_str(), fr.line, fr.func.c_str());
      if (i == 2 && n > 4) {
        fprintf(stderr, "  ... %zu frames omitted ...\n", n - 4);
        i = n - 2; // next loop prints leaf (#n-1)
      }
    }
  }
}


// ------------------- globals -------------------
static std::mutex g_mutex;

static std::atomic<bool> g_callbacks_enabled{false};  // set true if subscribe() succeeds
static CUpti_SubscriberHandle g_sub = nullptr;
static bool g_started = false;

static std::unordered_map<StreamKey,  std::vector<Frame>, KeyHash, KeyEq>  g_stream2stack; // pointer map
static std::unordered_map<StreamIdKey,std::vector<Frame>, SidHash, SidEq>  g_sid2stack;    // activity streamId map

static std::unordered_map<uint32_t, std::vector<Frame>> g_corr2stack_driver;
static std::unordered_map<uint32_t, std::vector<Frame>> g_corr2stack_runtime;

// Map CUDA Graph replay launches to their Python stacks.
// Used when kernels executed from a graph share the graph-launch corrId.
static std::unordered_map<uint32_t, std::vector<Frame>> g_corr2stack_graph;


static FILE* g_out = nullptr;          // file sink (optional)
static bool  g_mem_mode = false;       // memory sink on/off
static std::deque<Event> g_queue;      // memory queue
static size_t g_queue_max = 200000;    // capacity
static uint64_t g_q_dropped = 0;       // drop counter

static std::atomic<int> g_capture_py_stack{1}; // toggle Python stack capture in callbacks

static bool do_start_internal(const char* path, bool mem_mode);
static void do_stop_internal();
// static void maybe_autostart_from_env();
static void on_py_atexit(void);

// Minimal runtime param shapes for CUDA Graph capture/replay
struct RT_cudaStreamBeginCapture_params { void* stream; int mode; };
struct RT_cudaStreamEndCapture_params   { void* stream; void** pGraph; };
struct RT_cudaGraphInstantiate_params   { void** pGraphExec; void* graph; void* errorNode; unsigned long long flags; };
struct RT_cudaGraphLaunch_params        { void* graphExec; void* stream; };


// Dispatch snapshots are copied while Python owns the GIL. CUDA callbacks on
// this thread can read them without acquiring Python or a process-wide lock.
struct StackTag {
  uint64_t id;
  int device;
  CUstream stream;
  std::vector<Frame> stack;
};
static thread_local std::vector<StackTag> t_stack_tags;
static std::atomic<uint64_t> g_next_tag{1};

// ------------------- utilities -------------------
static bool get_stream_id_for(CUstream h, uint32_t* out_sid, CUcontext ctx = nullptr) {
#if defined(CUPTI_API_VERSION) && (CUPTI_API_VERSION >= 13)  // CUDA 11+
  CUptiResult r = cuptiGetStreamId(ctx, h, out_sid);  // maps CUstream -> activity streamId
  return r == CUPTI_SUCCESS;
#else
  return false;  // old CUPTI: no mapping available
#endif
}

// Never acquire the GIL from a CUDA callback: the caller may hold a CUDA /
// framework lock that another Python thread is waiting for while owning the GIL.
// Prefer the dispatch snapshot, including when PyTorch released the GIL.
static std::vector<Frame> get_python_stack() {
  std::vector<Frame> out;  // we’ll return root->leaf
  if (!t_stack_tags.empty()) return t_stack_tags.back().stack;
  if (!Py_IsInitialized() || !PyGILState_Check()) return out;
  // Preserve an exception already pending at API EXIT.
  PyObject *exc_type, *exc_value, *exc_tb;
  PyErr_Fetch(&exc_type, &exc_value, &exc_tb);
  PyThreadState* tstate = PyThreadState_Get();
  if (tstate) {
#if PY_VERSION_HEX >= 0x030B0000
    PyObject* f = (PyObject*)PyThreadState_GetFrame(tstate);  // new ref
#else
    PyObject* f = (PyObject*)tstate->frame;                   // borrowed
    Py_XINCREF(f);
#endif
    while (f) {
      int lineno = PyFrame_GetLineNumber((PyFrameObject*)f);
      PyObject* co = (PyObject*)PyFrame_GetCode((PyFrameObject*)f); // new ref
      std::string file_str = "<unknown>", name_str = "<unknown>";
      if (co) {
        PyObject* fn = PyObject_GetAttrString(co, "co_filename");
        PyObject* nm = PyObject_GetAttrString(co, "co_name");
        if (fn && PyUnicode_Check(fn)) if (const char* s = PyUnicode_AsUTF8(fn)) file_str.assign(s);
        if (nm && PyUnicode_Check(nm)) if (const char* s = PyUnicode_AsUTF8(nm)) name_str.assign(s);
        Py_XDECREF(fn); Py_XDECREF(nm); Py_DECREF(co);
      }
      out.push_back(Frame{file_str, lineno, name_str});
      PyObject* prev = (PyObject*)PyFrame_GetBack((PyFrameObject*)f); // new ref
      Py_DECREF(f);
      f = prev;
    }
  }
  PyErr_Restore(exc_type, exc_value, exc_tb);
  std::reverse(out.begin(), out.end());   // normalize to root→leaf
  return out;
}

// ------------------- CUPTI callbacks -------------------

#define CUDA_LAUNCH_CBID  CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000
typedef cudaLaunchKernel_v7000_params cudaLaunchParamsT;

struct DRV_cuGraphLaunch_exec { CUgraphExec hGraphExec; CUstream hStream; };
struct DRV_cuGraphLaunch_graph{ CUgraph     hGraph;     CUstream hStream; };



static void CUPTIAPI api_cb(void*, CUpti_CallbackDomain domain,
                            CUpti_CallbackId cbid, const CUpti_CallbackData* info) {
  const bool enter = (info->callbackSite == CUPTI_API_ENTER);
  const bool exit  = (info->callbackSite == CUPTI_API_EXIT);
  if (!enter && !exit) return;

  const bool want_py_stack = g_capture_py_stack.load(std::memory_order_relaxed);

  // ---------------- DRIVER API (safe subset) ----------------
  if (domain == CUPTI_CB_DOMAIN_DRIVER_API) {
    switch (cbid) {
      case CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCapture: {
        if (!enter) break;
        auto* p = reinterpret_cast<const cuStreamBeginCapture_params*>(info->functionParams);
        if (!p) return;
        CUstream h = p->hStream;
        {
          std::lock_guard<std::mutex> lk(g_mutex);
          auto& cs = g_capture_by_stream[h];
          cs.active = true;
          cs.node_stacks.clear();
        }
        if (want_py_stack && dbg_on()) {
          std::vector<Frame> st = get_python_stack();
          dbg_dump_stack(st, "GraphLaunch(DRV ENTER)");
        }

        return;
      }
      case CUPTI_DRIVER_TRACE_CBID_cuStreamEndCapture: {
        if (!exit) break;
        auto* p = (const cuStreamEndCapture_params*)info->functionParams;
        CUstream h = p ? p->hStream : nullptr;
        CUgraph  g = (p && p->phGraph) ? *p->phGraph : nullptr;

        size_t n = 0;
        if (h && g) {
          std::lock_guard<std::mutex> lk(g_mutex);
          auto it = g_capture_by_stream.find(h);
          if (it != g_capture_by_stream.end()) {
            n = it->second.node_stacks.size();
            // Only set if we don't already have data (idempotent)
            auto ins = g_graph_to_node_stacks.emplace(g, std::move(it->second.node_stacks));
            if (!ins.second && !ins.first->second.empty()) {
              // already populated by the other domain; leave it alone
              n = ins.first->second.size();
            }
            it->second.active = false;
            it->second.node_stacks.clear();
          }
        }
        // Dump captured node stacks with line numbers
        if (dbg_on() && g) {
        std::lock_guard<std::mutex> lk(g_mutex);
          auto it2 = g_graph_to_node_stacks.find(g);
          if (it2 != g_graph_to_node_stacks.end()) {
            const auto& stacks = it2->second;
            for (size_t i = 0; i < stacks.size(); ++i) {
              char tag[160];
              snprintf(tag, sizeof(tag), "CAPTURED node[%zu]", i);
              dbg_dump_stack(stacks[i], tag);
            }
          }
        }

        DBG("EndCapture(DRV EXIT) stream=%p graph=%p nodes=%zu", (void*)h, (void*)g, n);
        return;
      }

      case CUPTI_DRIVER_TRACE_CBID_cuGraphInstantiate: {
        if (!exit) break;  // read outputs on EXIT
        auto* p = reinterpret_cast<const cuGraphInstantiate_params*>(info->functionParams);
        if (!p) return;
        CUgraphExec exec = (p->phGraphExec ? *p->phGraphExec : nullptr);
        CUgraph     g    = p->hGraph;
        if (exec && g) {
          std::lock_guard<std::mutex> lk(g_mutex);
          g_exec_to_graph[exec] = g;
        }
        if (dbg_on()) {
          std::vector<Frame> st = get_python_stack();
          dbg_dump_stack(st, "INSTANTIATE(DRV EXIT)");
        }
        return;
      }
      case CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch: {
        // Driver param layout varies across CUDA versions; do not dereference.
        if (!enter) break;
        DBG("GraphLaunch(DRV ENTER) (skipping params due to ABI variance)");
        if (want_py_stack) {
          auto st = get_python_stack();
          std::lock_guard<std::mutex> lk(g_mutex);
          g_corr2stack_graph[info->correlationId] = std::move(st);
        }
        return;
      }
      case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel: {
        if (!enter) break;
        if (!info->functionParams) return;
        std::vector<Frame> stack;
        if (want_py_stack) stack = get_python_stack();
        auto* p = reinterpret_cast<const cuLaunchKernel_params*>(info->functionParams);
        CUstream h = p ? p->hStream : nullptr;
        std::vector<Frame> copy = stack;
        {
          std::lock_guard<std::mutex> lk(g_mutex);
          g_corr2stack_driver[info->correlationId] = std::move(stack);
          auto it = g_capture_by_stream.find(h);
          if (h && it != g_capture_by_stream.end() && it->second.active) {
            it->second.node_stacks.push_back(std::move(copy));
            {
              auto& ns = it->second.node_stacks;
              size_t idx = ns.size() - 1;
              char tag[160];
              snprintf(tag, sizeof(tag), "CAPTURE node[%zu] (append)", idx);
              dbg_dump_stack(ns[idx], tag);
            }
          }
        }
        return;
      }
      default: break;
    }
  }

  // ---------------- RUNTIME API (primary path) ----------------
  if (domain == CUPTI_CB_DOMAIN_RUNTIME_API) {
    if (!info->functionName) return;
    const char* fn = info->functionName;

    // BeginCapture (ENTER)
    if (enter && strncmp(fn, "cudaStreamBeginCapture", 22) == 0) {
      auto* p = reinterpret_cast<const RT_cudaStreamBeginCapture_params*>(info->functionParams);
      if (!p) return;
      CUstream h = (CUstream)p->stream;
      {
        std::lock_guard<std::mutex> lk(g_mutex);
        auto& cs = g_capture_by_stream[h];
        cs.active = true;
        cs.node_stacks.clear();
      }
      DBG("BeginCapture(RT ENTER) stream=%p", (void*)h);
      return;
    }

    // EndCapture (EXIT: read *pGraph)
    if (exit && strncmp(fn, "cudaStreamEndCapture", 20) == 0) {
      auto* p = (const RT_cudaStreamEndCapture_params*)info->functionParams;
      CUstream h = p ? (CUstream)p->stream : nullptr;
      CUgraph  g = (p && p->pGraph) ? (CUgraph)(*p->pGraph) : nullptr;

      size_t n = 0;
      if (h && g) {
        std::lock_guard<std::mutex> lk(g_mutex);
        auto it = g_capture_by_stream.find(h);
        if (it != g_capture_by_stream.end()) {
          n = it->second.node_stacks.size();
          // Only set if not already populated by the driver handler
          auto ins = g_graph_to_node_stacks.emplace(g, std::move(it->second.node_stacks));
          if (!ins.second && !ins.first->second.empty()) {
            n = ins.first->second.size(); // keep existing
          }
          it->second.active = false;
          it->second.node_stacks.clear();
        }
      }
      if (dbg_on() && g) {
        std::lock_guard<std::mutex> lk(g_mutex);
        auto it2 = g_graph_to_node_stacks.find(g);
        if (it2 != g_graph_to_node_stacks.end()) {
          const auto& stacks = it2->second;
          for (size_t i = 0; i < stacks.size(); ++i) {
            char tag[160];
            snprintf(tag, sizeof(tag), "CAPTURED node[%zu]", i);
            dbg_dump_stack(stacks[i], tag);
          }
        }
      }
      DBG("EndCapture(RT EXIT) stream=%p graph=%p nodes=%zu", (void*)h, (void*)g, n);
      return;
    }

    // Instantiate (EXIT: read *pGraphExec)
    if (exit && strncmp(fn, "cudaGraphInstantiate", 20) == 0) {
      auto* p = reinterpret_cast<const RT_cudaGraphInstantiate_params*>(info->functionParams);
      if (!p) return;
      CUgraphExec exec = p->pGraphExec ? (CUgraphExec)(*p->pGraphExec) : nullptr;
      CUgraph     g    = (CUgraph)p->graph;
      if (exec && g) {
        std::lock_guard<std::mutex> lk(g_mutex);
        g_exec_to_graph[exec] = g;
      }
      if (dbg_on()) {
        std::vector<Frame> st = get_python_stack();
        dbg_dump_stack(st, "INSTANTIATE(RT EXIT)");
      }

      return;
    }

    // GraphLaunch (ENTER: primary arming; allow default stream 0)
    if (enter && strncmp(fn, "cudaGraphLaunch", 15) == 0) {
      auto* p = (const RT_cudaGraphLaunch_params*)info->functionParams;
      CUgraphExec exec = p ? (CUgraphExec)p->graphExec : nullptr;
      CUstream    h    = p ? (CUstream)p->stream     : nullptr;

      uint32_t sid = 0;

      uint32_t device_id = 0;
      const bool have_device = cuptiGetDeviceId(info->context, &device_id) == CUPTI_SUCCESS;
      const int dev = (int)device_id;
      if (have_device && get_stream_id_for(h, &sid, info->context)) {
        std::lock_guard<std::mutex> lk(g_mutex);
        StreamIdKey key{dev, sid};
        // reset sticky from previous replay on this stream (fresh window)
        g_last_stack_by_sid.erase(key);
        g_replay_by_sid[key] = ReplayProgress{exec, /*graph*/nullptr, /*next_idx*/0};
      }

      if (dbg_on()) {
        std::vector<Frame> st = get_python_stack();
        dbg_dump_stack(st, "INSTANTIATE(DRV EXIT)");
      }


      if (want_py_stack) {
        std::vector<Frame> stack = get_python_stack();
        std::lock_guard<std::mutex> lk(g_mutex);
        g_corr2stack_graph[info->correlationId] = std::move(stack);
      }
      return;
    }


    // Regular kernel launch (ENTER)
    if (enter && strncmp(fn, "cudaLaunchKernel", 16) == 0) {
      if (!info->functionParams) return;
      std::vector<Frame> stack;
      if (want_py_stack) stack = get_python_stack();
      {
        std::lock_guard<std::mutex> lk(g_mutex);
        g_corr2stack_runtime[info->correlationId] = stack;
      }
      auto* pp = reinterpret_cast<const cudaLaunchParamsT*>(info->functionParams);
      CUstream h = pp ? (CUstream)pp->stream : nullptr;
      if (h) {
        std::lock_guard<std::mutex> lk(g_mutex);
        auto it = g_capture_by_stream.find(h);
        if (it != g_capture_by_stream.end() && it->second.active) {
          it->second.node_stacks.push_back(std::move(stack));
          {
            auto& ns = it->second.node_stacks;
            size_t idx = ns.size() - 1;
            char tag[160];
            snprintf(tag, sizeof(tag), "CAPTURE node[%zu] (append)", idx);
            dbg_dump_stack(ns[idx], tag);
          }
        }
      }
      return;
    }
  }
}




// ------------------- Activity buffer handlers -------------------
static void CUPTIAPI buf_req(uint8_t** buffer, size_t* size, size_t* maxNumRecords) {
  *size = 1 << 20; // 1MB
  *buffer = (uint8_t*) malloc(*size);
  *maxNumRecords = 0;
}

static void enqueue_event_mem(const CUpti_ActivityKernel8* k, const std::vector<Frame>* stack) {
  Event e;
  e.kernel = s_strn(k->name);
  e.correlation = k->correlationId;
  e.device = k->deviceId;
  e.stream  = k->streamId;
  e.start   = k->start;
  e.end     = k->end;
  if (stack) e.stack = *stack;
  std::lock_guard<std::mutex> lock(g_mutex);
  if (g_queue.size() >= g_queue_max) { g_queue.pop_front(); ++g_q_dropped; }
  g_queue.emplace_back(std::move(e));
}

static void dump_kernel_record_to_file(const CUpti_ActivityKernel8* k, const std::vector<Frame>* stack) {
  if (!g_out) return;
  const std::string kname = s_strn(k->name);
  fprintf(g_out, "{");
  fprintf(g_out, "\"kernel\":\"%s\"", jesc(kname).c_str());
  fprintf(g_out, ",\"correlation\":%u", k->correlationId);
  fprintf(g_out, ",\"device\":%u,\"stream\":%u", k->deviceId, k->streamId);
  fprintf(g_out, ",\"start\":%" PRIu64 ",\"end\":%" PRIu64, k->start, k->end);
  fprintf(g_out, ",\"stack\":[");
  if (stack){
    bool first=true;
    for (auto &fr : *stack){
      if (!first) fprintf(g_out, ",");
      first=false;
      fprintf(g_out, "[\"%s\",%d,\"%s\"]",
        jesc(fr.file).c_str(), fr.line, jesc(fr.func).c_str());
    }
  }
  fprintf(g_out, "]}");
  fputc('\n', g_out);
}

static void CUPTIAPI buf_done(CUcontext ctx, uint32_t streamId,
                              uint8_t* buffer, size_t size, size_t validSize) {
  if (validSize == 0) { free(buffer); return; }

  CUpti_Activity* rec = nullptr;
  CUptiResult status = CUPTI_SUCCESS;

  do {
    status = cuptiActivityGetNextRecord(buffer, validSize, &rec);
    if (status != CUPTI_SUCCESS || !rec) break;

    if (rec->kind != CUPTI_ACTIVITY_KIND_KERNEL &&
        rec->kind != CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
      continue;
    }

    const CUpti_ActivityKernel8* k =
        reinterpret_cast<const CUpti_ActivityKernel8*>(rec);

    // Copy the matched stack under lock
    std::vector<Frame> local_stack;
    bool used_runtime = false, used_driver = false, used_graph = false;

    // 0) Prefer node-level mapping by (device, streamId) during graph replay
    {
      std::lock_guard<std::mutex> lock(g_mutex);
      StreamIdKey key{(int)k->deviceId, k->streamId};
      auto it = g_replay_by_sid.find(key);
      if (it != g_replay_by_sid.end()) {
        ReplayProgress rp = it->second;

        // Resolve which graph this replay belongs to
        CUgraph g = nullptr;
        if (rp.exec) {
          auto gx = g_exec_to_graph.find(rp.exec);
          if (gx != g_exec_to_graph.end()) g = gx->second;
        } else if (rp.graph) {
          g = rp.graph;
        }

        if (g) {
          auto ns = g_graph_to_node_stacks.find(g);
          if (ns != g_graph_to_node_stacks.end()) {
            const auto& stacks = ns->second;
            if (rp.next_idx < stacks.size()) {
              local_stack = stacks[rp.next_idx++];        // assign captured op site
              g_replay_by_sid[key].next_idx = rp.next_idx; // write back progress

              // remember this node’s stack for subsequent kernels from the same node
              g_last_stack_by_sid[key] = local_stack;

              DBG("JOIN(node) sid=%u idx=%zu kernel=%s",
                  k->streamId, rp.next_idx - 1, s_strn(k->name).c_str());

              if (rp.next_idx >= stacks.size()) {
                // end of this replay window — clear sticky state
                g_replay_by_sid.erase(key);
                // g_last_stack_by_sid.erase(key);
              }
            } else {
              DBG("MISS(node): idx=%zu / size=%zu", rp.next_idx, stacks.size());
            }

          } else {
            DBG("MISS(node): graph has no node_stacks");
          }
        } else {
          DBG("MISS(node): cannot resolve graph from exec/graph");
        }
      }
    }


    // 1) Only if node-level mapping didn't set a stack, try correlationId maps
    {
      std::lock_guard<std::mutex> lock(g_mutex);

      if (local_stack.empty() && g_callbacks_enabled) {
        auto itR = g_corr2stack_runtime.find(k->correlationId);
        if (itR != g_corr2stack_runtime.end()) {
          local_stack = itR->second;
          used_runtime = true;
        } else {
          auto itD = g_corr2stack_driver.find(k->correlationId);
          if (itD != g_corr2stack_driver.end()) {
            local_stack = itD->second;
            used_driver = true;
          }
        }
      }

      // 2) Only if still empty, allow macro graph corrId (replay site)
      if (local_stack.empty() && g_callbacks_enabled) {
        auto itG = g_corr2stack_graph.find(k->correlationId);
        if (itG != g_corr2stack_graph.end()) {
          local_stack = itG->second;
          used_graph = true;
        }
      }

      // StreamId tag fallback
      if (local_stack.empty()) {
        auto itS = g_sid2stack.find(StreamIdKey{(int)k->deviceId, k->streamId});
        if (itS != g_sid2stack.end()) {
          local_stack = itS->second;
        }
      }
      // Final fallback: reuse last node’s stack for multi-kernel nodes (cuBLAS/cuDNN)
      if (local_stack.empty()) {
        auto itL = g_last_stack_by_sid.find(StreamIdKey{(int)k->deviceId, k->streamId});
        if (itL != g_last_stack_by_sid.end()) {
          local_stack = itL->second;
        }
      }

      if (g_callbacks_enabled) {
        if (used_runtime) g_corr2stack_runtime.erase(k->correlationId);
        if (used_driver)  g_corr2stack_driver.erase(k->correlationId);
        if (used_graph)   g_corr2stack_graph.erase(k->correlationId);
      }
    }

    // Emit using the copy (no locks)
    if (g_mem_mode) enqueue_event_mem(k, local_stack.empty() ? nullptr : &local_stack);
    else            dump_kernel_record_to_file(k, local_stack.empty() ? nullptr : &local_stack);

  } while (status == CUPTI_SUCCESS);

  size_t dropped = 0;
  cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped);
  free(buffer);
}


// ------------------- Python API -------------------
static const char* cupti_errstr(CUptiResult r){
  const char* s = nullptr;
  if (cuptiGetResultString(r, &s) == CUPTI_SUCCESS && s) return s;
  return "<unknown>";
}

// helper: enable one callback and log the result
static void enable_cb_or_log(CUpti_SubscriberHandle sub,
                             CUpti_CallbackDomain dom,
                             CUpti_CallbackId cbid,
                             const char* name) {
  CUptiResult r = cuptiEnableCallback(1, sub, dom, cbid);
  if (r != CUPTI_SUCCESS) {
    const char* rs = nullptr;
    cuptiGetResultString(r, &rs);
    fprintf(stderr, "[opguard_cupti] cuptiEnableCallback(%s) failed: %d (%s)\n",
            name, (int)r, rs ? rs : "<unknown>");
  } else {
    DBG("Enabled %s", name);
  }
}

static bool do_start_internal(const char* path, bool mem_mode) {
  if (g_started) return true;
  DBG("start_internal: path=%s mem_mode=%d", path?path:"<null>", (int)mem_mode);

  const char* stacks = getenv("OPGUARD_STACKS");
  if (stacks) g_capture_py_stack.store(atoi(stacks) != 0, std::memory_order_relaxed);

  if (!mem_mode) {
    g_out = fopen(path, "w");
    if (!g_out) return false;
  } else {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_queue.clear();
    g_queue_max = 200000;
    g_q_dropped = 0;
  }

  CUptiResult r = cuptiSubscribe(&g_sub, (CUpti_CallbackFunc)api_cb, nullptr);
  g_callbacks_enabled = (r == CUPTI_SUCCESS);
  if (!g_callbacks_enabled) {
    const char* rs = nullptr;
    cuptiGetResultString(r, &rs);
    fprintf(stderr, "[opguard_cupti] cuptiSubscribe failed: %d (%s)\n", (int)r, rs?rs:"?");
    return false;
  }

  // ======= DRIVER API: enable only the CBIDs we use =========
  // Do NOT enable the entire driver domain; this avoids cuModuleGetLoadingMode etc.
  enable_cb_or_log(g_sub, CUPTI_CB_DOMAIN_DRIVER_API,
                   CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel, "DRIVER cuLaunchKernel");
  enable_cb_or_log(g_sub, CUPTI_CB_DOMAIN_DRIVER_API,
                   CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCapture, "DRIVER cuStreamBeginCapture");
  enable_cb_or_log(g_sub, CUPTI_CB_DOMAIN_DRIVER_API,
                   CUPTI_DRIVER_TRACE_CBID_cuStreamEndCapture, "DRIVER cuStreamEndCapture");
  enable_cb_or_log(g_sub, CUPTI_CB_DOMAIN_DRIVER_API,
                   CUPTI_DRIVER_TRACE_CBID_cuGraphInstantiate, "DRIVER cuGraphInstantiate");
  enable_cb_or_log(g_sub, CUPTI_CB_DOMAIN_DRIVER_API,
                   CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch, "DRIVER cuGraphLaunch");

  // ======= RUNTIME API: either selective, or whole domain + name filter =======
  // Easiest: enable the whole runtime domain and let api_cb filter by functionName
  // (safer across versioned CBIDs like cudaLaunchKernel_v7000 etc.)
  CUptiResult rr = cuptiEnableDomain(1, g_sub, CUPTI_CB_DOMAIN_RUNTIME_API);
  if (rr != CUPTI_SUCCESS) {
    const char* rs = nullptr;
    cuptiGetResultString(rr, &rs);
    fprintf(stderr, "[opguard_cupti] cuptiEnableDomain(RUNTIME) failed: %d (%s)\n", (int)rr, rs?rs:"?");
  } else {
    DBG("Enabled RUNTIME domain (name-filtered in api_cb)");
  }

  // ======= Activity kinds =======
  CUptiResult ra = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL);
  CUptiResult rb = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
  fprintf(stderr, "[opguard_cupti] activityEnable KERNEL=%d (%s) CONC_KERNEL=%d (%s)\n",
          (int)ra, cupti_errstr(ra), (int)rb, cupti_errstr(rb));
  cuptiActivityRegisterCallbacks(buf_req, buf_done);

  g_mem_mode = mem_mode;
  g_started = true;
  return true;
}

static void do_stop_internal() {
  if (!g_started) return;
  cuptiActivityFlushAll(0);
  if (g_callbacks_enabled && g_sub) { cuptiUnsubscribe(g_sub); g_sub = nullptr; }
  g_callbacks_enabled = false;
  if (g_out) { fclose(g_out); g_out = nullptr; }
  g_started = false;
}

static PyObject* py_start(PyObject*, PyObject* args) {
  const char* path = nullptr;
  if (!PyArg_ParseTuple(args, "s", &path)) return nullptr;
  if (!do_start_internal(path, /*mem_mode=*/false))
    return PyErr_Format(PyExc_RuntimeError, "failed to start CUPTI to %s", path);
  Py_RETURN_NONE;
}

static PyObject* py_start_mem(PyObject*, PyObject* args) {
  int limit = 200000;
  if (!PyArg_ParseTuple(args, "|i", &limit)) return nullptr;
  if (!do_start_internal(/*unused*/"/dev/null", /*mem_mode=*/true))
    return PyErr_Format(PyExc_RuntimeError, "failed to start CUPTI mem mode");
  // keep your limit handling if you want it (g_queue_max)
  if (limit > 0) { std::lock_guard<std::mutex> lk(g_mutex); g_queue_max = (size_t)limit; }
  Py_RETURN_NONE;
}

static PyObject* py_stop(PyObject*, PyObject*) { do_stop_internal(); Py_RETURN_NONE; }


static PyObject* py_drain(PyObject*, PyObject* args) {
  int max_items = 5000;
  if (!PyArg_ParseTuple(args, "|i", &max_items)) return nullptr;
  if (max_items <= 0) max_items = 5000;

  std::deque<Event> local;
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    size_t n = std::min((size_t)max_items, g_queue.size());
    for (size_t i = 0; i < n; ++i) {
      local.emplace_back(std::move(g_queue.front()));
      g_queue.pop_front();
    }
  }

  PyObject* list = PyList_New(local.size());
  size_t idx = 0;
  for (auto &e : local) {
    PyObject* py_stack = PyList_New(e.stack.size());
    for (size_t i = 0; i < e.stack.size(); ++i) {
      const auto& fr = e.stack[i];
      PyObject* triple = PyList_New(3);
      PyList_SET_ITEM(triple, 0, PyUnicode_FromString(fr.file.c_str()));
      PyList_SET_ITEM(triple, 1, PyLong_FromLong(fr.line));
      PyList_SET_ITEM(triple, 2, PyUnicode_FromString(fr.func.c_str()));
      PyList_SET_ITEM(py_stack, i, triple);
    }
    PyObject* d = PyDict_New();
    PyDict_SetItemString(d, "kernel", PyUnicode_FromString(e.kernel.c_str()));
    PyDict_SetItemString(d, "correlation", PyLong_FromUnsignedLong(e.correlation));
    PyDict_SetItemString(d, "device", PyLong_FromUnsignedLong(e.device));
    PyDict_SetItemString(d, "stream", PyLong_FromUnsignedLong(e.stream));
    PyDict_SetItemString(d, "start", PyLong_FromUnsignedLongLong(e.start));
    PyDict_SetItemString(d, "end", PyLong_FromUnsignedLongLong(e.end));
    PyDict_SetItemString(d, "stack", py_stack);
    Py_DECREF(py_stack);
    PyList_SET_ITEM(list, idx++, d);
  }
  return list;
}
static PyObject* py_flush(PyObject*, PyObject*) {
  // Ask CUPTI to deliver all activity records now
  cuptiActivityFlushAll(0);
  if (g_out) fflush(g_out);   // file-sink case
  Py_RETURN_NONE;
}

static PyObject* py_tag_stream(PyObject*, PyObject* args) {
  int device = 0;
  unsigned long long stream_ull = 0;
  PyObject* py_stack = nullptr;
  if (!PyArg_ParseTuple(args, "iKO", &device, &stream_ull, &py_stack)) return nullptr;
  if (!PyList_Check(py_stack)) return PyErr_Format(PyExc_TypeError, "stack must be a list");

  std::vector<Frame> stack;
  const Py_ssize_t n = PyList_Size(py_stack);
  stack.reserve((size_t)n);
  for (Py_ssize_t i = 0; i < n; ++i) {
    PyObject* t = PyList_GetItem(py_stack, i);
    if (!PyList_Check(t) || PyList_Size(t) != 3) continue;
    const char* file = PyUnicode_AsUTF8(PyList_GetItem(t, 0));
    int line = (int)PyLong_AsLong(PyList_GetItem(t, 1));
    const char* func = PyUnicode_AsUTF8(PyList_GetItem(t, 2));
    stack.push_back(Frame{ file ? file : "<unknown>", line, func ? func : "<unknown>" });
  }

  StreamKey pkey{ device, (CUstream)(uintptr_t)stream_ull };
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_stream2stack[pkey] = stack;  // pointer map (used in callback fast path)
  }

  uint32_t sid = 0;
  if (get_stream_id_for(pkey.h, &sid)) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_sid2stack[{device, sid}] = std::move(stack);
  }
  Py_RETURN_NONE;
}

// Matched by trace.py's dispatch try/finally. No CUDA calls here: a tag
// describes the launching host scope, and correlation IDs carry it to activity.
static PyObject* py_tag_stream_push(PyObject*, PyObject* args) {
  int device;
  unsigned long long stream;
  PyObject* py_stack;
  const char* op = nullptr;
  if (!PyArg_ParseTuple(args, "iKO|s", &device, &stream, &py_stack, &op)) return nullptr;
  PyObject* seq = PySequence_Fast(py_stack, "stack must be a sequence");
  if (!seq) return nullptr;
  std::vector<Frame> stack;
  for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(seq); ++i) {
    PyObject* row = PySequence_Fast(PySequence_Fast_GET_ITEM(seq, i), "frame must be a sequence");
    if (!row) { Py_DECREF(seq); return nullptr; }
    if (PySequence_Fast_GET_SIZE(row) != 3) {
      Py_DECREF(row); Py_DECREF(seq);
      return PyErr_Format(PyExc_ValueError, "frame must contain file, line, function");
    }
    const char* file = PyUnicode_AsUTF8(PySequence_Fast_GET_ITEM(row, 0));
    long line = PyLong_AsLong(PySequence_Fast_GET_ITEM(row, 1));
    const char* func = PyUnicode_AsUTF8(PySequence_Fast_GET_ITEM(row, 2));
    if (!file || !func || PyErr_Occurred()) { Py_DECREF(row); Py_DECREF(seq); return nullptr; }
    stack.push_back(Frame{file, (int)line, func});
    Py_DECREF(row);
  }
  Py_DECREF(seq);
  uint64_t id = g_next_tag.fetch_add(1, std::memory_order_relaxed);
  PyObject* result = PyLong_FromUnsignedLongLong(id);
  if (!result) return nullptr;
  t_stack_tags.push_back(StackTag{id, device, (CUstream)(uintptr_t)stream, std::move(stack)});
  return result;
}

static PyObject* py_tag_stream_pop(PyObject*, PyObject* args) {
  int device;
  unsigned long long stream, id;
  if (!PyArg_ParseTuple(args, "iKK", &device, &stream, &id)) return nullptr;
  if (t_stack_tags.empty() || t_stack_tags.back().id != id ||
      t_stack_tags.back().device != device ||
      t_stack_tags.back().stream != (CUstream)(uintptr_t)stream)
    return PyErr_Format(PyExc_ValueError, "stream tags must be popped on their owner thread in LIFO order");
  t_stack_tags.pop_back();
  Py_RETURN_NONE;
}

static PyObject* py_set_py_stack_capture(PyObject*, PyObject* args) {
  int on = 1;
  if (!PyArg_ParseTuple(args, "|p", &on)) return nullptr;
  g_capture_py_stack.store(on ? 1 : 0, std::memory_order_relaxed);
  Py_RETURN_NONE;
}


static PyMethodDef Methods[] = {
  {"start",      (PyCFunction)py_start,      METH_VARARGS, "Start CUPTI tracing to a file"},
  {"start_mem",  (PyCFunction)py_start_mem,  METH_VARARGS, "Start CUPTI tracing in memory (optional buffer limit)"},
  {"drain",      (PyCFunction)py_drain,      METH_VARARGS, "Drain up to N events from memory queue -> list[dict]"},
  {"tag_stream_push", (PyCFunction)py_tag_stream_push, METH_VARARGS, "Push a host-thread dispatch stack snapshot"},
  {"tag_stream_pop", (PyCFunction)py_tag_stream_pop, METH_VARARGS, "Pop a host-thread dispatch stack snapshot"},
  {"tag_stream", (PyCFunction)py_tag_stream, METH_VARARGS, "Tag (device, stream) with a Python stack"},
  {"stop",       (PyCFunction)py_stop,       METH_NOARGS,  "Stop CUPTI tracing"},
  {"flush", (PyCFunction)py_flush, METH_NOARGS, "Flush CUPTI activity buffers and file"},
  {"set_py_stack_capture", (PyCFunction)py_set_py_stack_capture, METH_VARARGS, "Enable/disable Python stack capture in callbacks"},
  {nullptr, nullptr, 0, nullptr}
};

static struct PyModuleDef mod = {
  PyModuleDef_HEAD_INIT, "opguard_cupti", "CUPTI bridge", -1, Methods
};


static void on_py_atexit(void) { do_stop_internal(); }

PyMODINIT_FUNC PyInit_opguard_cupti(void) {
  PyObject* m = PyModule_Create(&mod);
  if (m) {
    Py_AtExit(on_py_atexit);  // flush on interpreter exit
  }
  return m;
}

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <string>
#include <iostream>

#include <Python.h>
#include <frameobject.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cublasLt.h>

#include <iomanip>
#include <unordered_map>
#include <vector>
#include <tuple>
#include <set>
#include <mutex>
#include <atomic>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <unordered_set>
#include <algorithm>  // for std::reverse
#include <sys/utsname.h>
#include <ctime>
#include <cstdint>
#include <deque>
#include <thread>
#include <chrono>

// ===================== Identity / basics =====================
static std::atomic<uint64_t> g_evt_count{0};


// --- identity helpers ---
static inline pid_t self_pid() { return getpid(); }
static inline long  self_tid() { return syscall(SYS_gettid); }

// Called only from build_py_snapshot with the GIL held. Use the same identity
// resolver as Python so startup and initialized distributed runs agree.
static inline std::string detect_rank() {
  PyObject* mod = PyImport_ImportModule("opguard.distributed");
  PyObject* rank = mod ? PyObject_CallMethod(mod, "get_global_rank", nullptr) : nullptr;
  Py_XDECREF(mod);
  if (!rank) {
    PyErr_Clear();
    return "unknown";  // Never mislabel an unresolved multi-node rank as zero.
  }
  PyObject* value = PyObject_Str(rank);
  Py_DECREF(rank);
  const char* text = value ? PyUnicode_AsUTF8(value) : nullptr;
  std::string result = text ? text : "unknown";
  Py_XDECREF(value);
  if (PyErr_Occurred()) PyErr_Clear();
  return result;
}

static inline std::string hostname_short() {
  struct utsname u; if (uname(&u)==0) return std::string(u.nodename);
  return "unknown";
}

static inline std::string iso_time() {
  char buf[32]; std::time_t t=std::time(nullptr);
  std::tm tm; localtime_r(&t, &tm);
  std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm);
  return buf;
}

// mkdir -p helper
static inline void mkdir_p_for(const std::string& path) {
  size_t pos = path.rfind('/');
  if (pos == std::string::npos) return;
  std::string dir = path.substr(0, pos);
  if (dir.empty()) return;
  std::string cmd = "mkdir -p " + dir;
  ::system(cmd.c_str());
}

// ========= Env toggles =========
static inline bool env_silent() {
  static int v = []{ const char* e=getenv("PYHOOK_SILENT"); return (e && *e && strcmp(e,"0")!=0); }();
  return v;
}
static inline int env_stack_max() {
  static int v = []{ const char* e=getenv("PYHOOK_STACK_MAX"); return e? std::max(0, atoi(e)) : 0; }();
  return v;
}
static inline const char* env_stack_dir() {
  static const char* v = []{
    const char* e = getenv("PYHOOK_STACK_DIR");
    return (e && *e) ? e : "top";
  }();
  return v;
}

// ========= TLS reentrancy guard =========
static thread_local bool g_in_hook = false;
struct HookScope {
  bool entered=false;
  HookScope() { if (!g_in_hook) { g_in_hook=true; entered=true; } }
  ~HookScope(){ if (entered) g_in_hook=false; }
  explicit operator bool() const { return entered; }
};

// ========= Aggregation (per-site) =========
struct StackEntry {
  std::vector<std::tuple<std::string,int,std::string>> frames; // [[file, line, func], ...]
};
struct SiteBucket {
  uint64_t count = 0;                            // total events for this site
  std::vector<StackEntry> callstacks;            // unique stacks
  std::vector<uint64_t>   stack_counts;          // count per unique stack (aligned with callstacks)
  std::unordered_map<std::string, size_t> seen;  // key -> index into callstacks
  std::set<std::string> ops;
};
static std::mutex g_agg_mu;
static std::unordered_map<std::string, SiteBucket> g_by_site;
static std::atomic<bool> g_any_event{false};   // any event seen (process-wide)

// ========= Python sink (optional aggregated snapshot) =========
static inline int env_dump_every() {
  static int v = []{ const char* e=getenv("PYHOOK_DUMP_EVERY"); return e? std::max(0, atoi(e)) : 0; }();
  return v;  // 0 => only at exit
}
static inline const char* env_py_mod() {
  static const char* v = []{ const char* e=getenv("PYHOOK_PYMOD"); return (e && *e) ? e : "opguard_callsites_sink"; }();
  return v;
}
static inline const char* env_py_func() {
  static const char* v = []{ const char* e=getenv("PYHOOK_PYFUNC"); return (e && *e) ? e : "on_flush"; }();
  return v;
}
static std::atomic<bool> g_flush_in_progress{false};
static std::atomic<uint64_t> g_since_last_flush{0};
static PyObject* g_py_func = nullptr;

static void ensure_py_sink_resolved_once() {
  static std::atomic<bool> done{false};
  bool exp=false;
  if (!done.compare_exchange_strong(exp,true)) return;

  if (!Py_IsInitialized()) return;
  PyGILState_STATE gs = PyGILState_Ensure();
  PyObject* mod = PyImport_ImportModule(env_py_mod());  // new ref or NULL
  if (mod) {
    PyObject* func = PyObject_GetAttrString(mod, env_py_func()); // new ref
    Py_DECREF(mod);
    if (func && PyCallable_Check(func)) {
      g_py_func = func; // keep ref
    } else {
      Py_XDECREF(func);
    }
  }
  PyGILState_Release(gs);
}

static inline std::string frames_key_exact(
  const std::vector<std::tuple<std::string,int,std::string>>& frames) {
  std::string k; k.reserve(frames.size()*64);
  for (auto it = frames.rbegin(); it != frames.rend(); ++it) {
    const auto& fr = *it;
    k.append(std::get<0>(fr)); k.push_back(':');
    k.append(std::to_string(std::get<1>(fr))); k.push_back(':');
    k.append(std::get<2>(fr)); k.push_back('\n');
  }
  return k;
}

static PyObject* build_py_snapshot(bool final_flush) {
  if (!Py_IsInitialized() || !g_py_func) return nullptr;

  PyGILState_STATE gs = PyGILState_Ensure();

  PyObject* snap = PyDict_New();
  PyObject* meta = PyDict_New();
  PyObject* sites= PyDict_New();

  // meta
  PyDict_SetItemString(meta, "pid", PyLong_FromLong(self_pid()));
  PyDict_SetItemString(meta, "tid", PyLong_FromLong(self_tid()));
  PyDict_SetItemString(meta, "rank", PyUnicode_FromString(detect_rank().c_str()));
  PyDict_SetItemString(meta, "hostname", PyUnicode_FromString(hostname_short().c_str()));
  PyDict_SetItemString(meta, "time", PyUnicode_FromString(iso_time().c_str()));
  PyDict_SetItemString(meta, "final", PyBool_FromLong(final_flush ? 1 : 0));
  PyDict_SetItemString(snap, "meta", meta);  // transfers

  // sites
  {
    std::lock_guard<std::mutex> lk(g_agg_mu);
    for (const auto& kv : g_by_site) {
      const std::string& site = kv.first;
      const SiteBucket&  b    = kv.second;

      PyObject* sdict = PyDict_New();
      PyDict_SetItemString(sdict, "count", PyLong_FromUnsignedLongLong(b.count));

      PyObject* ops = PyList_New(0);
      for (const auto& op : b.ops) {
        PyList_Append(ops, PyUnicode_FromString(op.c_str()));
      }
      PyDict_SetItemString(sdict, "ops", ops);
      PyDict_SetItemString(sdict, "nstacks", PyLong_FromSize_t(b.callstacks.size()));

      PyDict_SetItem(sites, PyUnicode_FromString(site.c_str()), sdict);
    }
  }
  PyDict_SetItemString(snap, "sites", sites);

  PyGILState_Release(gs);
  return snap; // new reference
}

static void python_sink_flush(bool final_flush) {
  ensure_py_sink_resolved_once();
  if (!Py_IsInitialized() || !g_py_func) return;

  bool exp=false;
  if (!g_flush_in_progress.compare_exchange_strong(exp,true)) return;

  PyObject* snap = build_py_snapshot(final_flush);
  if (!snap) { g_flush_in_progress.store(false); return; }

  PyGILState_STATE gs = PyGILState_Ensure();
  PyObject* args = PyTuple_New(1);
  PyTuple_SET_ITEM(args, 0, snap);  // steals ref
  PyObject* ret = PyObject_CallObject(g_py_func, args);
  Py_DECREF(args);
  if (!ret) {
    static std::atomic<bool> warned{false};
    bool e=false; if (!warned.compare_exchange_strong(e,true)) { /* already warned */ }
    else { PyErr_Print(); }
  } else {
    Py_DECREF(ret);
  }
  PyGILState_Release(gs);
  g_flush_in_progress.store(false);
}

// ========= RF TLS bridge (optional, used if present) =========
static const char* (*RF_OP)()    = nullptr;
static const char* (*RF_STACK)() = nullptr;
static uint64_t    (*RF_TOK)()   = nullptr;

static void resolve_rf_syms_once() {
  static bool done = false;
  if (done) return;
  done = true;
  RF_OP    = (const char*(*)())dlsym(RTLD_DEFAULT, "rf_tls_op");
  RF_STACK = (const char*(*)())dlsym(RTLD_DEFAULT, "rf_tls_stack");
  RF_TOK   = (uint64_t(*)())    dlsym(RTLD_DEFAULT, "rf_tls_token");
}

// ========= Symbol resolution helpers =========
static void* dlopen_one(const char* so) {
  void* h = dlopen(so, RTLD_LAZY | RTLD_LOCAL | RTLD_NOLOAD);
  if (!h) h = dlopen(so, RTLD_LAZY | RTLD_LOCAL);
  return h;
}
static void* resolve_from_sos(const char* const* sos, const char* name) {
  void* p = dlsym(RTLD_NEXT, name);
  if (p) return p;
  for (const char* const* s = sos; *s; ++s) {
    if (void* h = dlopen_one(*s)) { p = dlsym(h, name); if (p) return p; }
  }
  return nullptr;
}
template <typename FP>
static FP resolve_any_of(const char* const* sos, std::initializer_list<const char*> names) {
  for (auto n : names) if (auto p = (FP)resolve_from_sos(sos, n)) return p;
  return nullptr;
}

// ========= Python stack capture =========
static std::pair<std::string, std::vector<std::tuple<std::string,int,std::string>>>
parse_stack_text(const std::string& text) {
  std::vector<std::tuple<std::string,int,std::string>> frames;
  frames.reserve(64);
  size_t pos = 0;
  while (pos < text.size()) {
    size_t nl = text.find('\n', pos);
    std::string line = text.substr(pos, (nl==std::string::npos) ? std::string::npos : nl-pos);
    pos = (nl==std::string::npos) ? text.size() : nl+1;
    if (line.empty()) continue;
    auto colon = line.rfind(':');
    auto inpos = line.find(" in ");
    if (colon != std::string::npos && inpos != std::string::npos && colon < inpos) {
      std::string file = line.substr(0, colon);
      int lineno = atoi(line.substr(colon+1, inpos-(colon+1)).c_str());
      std::string func = line.substr(inpos+4);
      frames.emplace_back(std::move(file), lineno, std::move(func));
    }
  }
  return {std::string(), std::move(frames)};
}
static inline std::string site_from_frames(const std::vector<std::tuple<std::string,int,std::string>>& frames) {
  if (frames.empty()) return "(no Python frames)";
  const auto& f = frames.front();
  // printf("first frame: %s:%d in %s\n", std::get<0>(f).c_str(), std::get<1>(f), std::get<2>(f).c_str());
  // printf("last frame: %s:%d in %s\n", std::get<0>(frames.back()).c_str(), std::get<1>(frames.back()), std::get<2>(frames.back()).c_str());
  // return first frame
  return std::get<0>(f) + std::string(":") + std::to_string(std::get<1>(f)) + " in " + std::get<2>(f);
}
static std::string capture_python_stack_impl() {
  std::string out;
  PyGILState_STATE gs = PyGILState_Ensure();
  PyThreadState* ts = PyThreadState_Get();
#if PY_VERSION_HEX >= 0x030B0000
  PyObject* f = (PyObject*)PyThreadState_GetFrame(ts);  // new ref
#else
  PyObject* f = ts ? (PyObject*)ts->frame : nullptr;    // borrowed
  Py_XINCREF(f);
#endif
  int max_frames = env_stack_max(), frames = 0;
  while (f && (max_frames<=0 || frames < max_frames)) {
    ++frames;
    int lineno = PyFrame_GetLineNumber((PyFrameObject*)f);
    PyCodeObject* co = PyFrame_GetCode((PyFrameObject*)f); // new ref
    if (co) {
      PyObject* fn = PyObject_GetAttrString((PyObject*)co, "co_filename");
      PyObject* nm = PyObject_GetAttrString((PyObject*)co, "co_name");
      const char* cfn = fn ? PyUnicode_AsUTF8(fn) : "<unknown>";
      const char* cnm = nm ? PyUnicode_AsUTF8(nm) : "<unknown>";
      if (cfn) out.append(cfn); else out.append("<bad>");
      out.push_back(':'); out.append(std::to_string(lineno));
      out.append(" in "); out.append(cnm ? cnm : "<bad>"); out.push_back('\n');
      Py_XDECREF(fn); Py_XDECREF(nm); Py_DECREF(co);
    }
#if PY_VERSION_HEX >= 0x030B0000
    PyObject* prev = (PyObject*)PyFrame_GetBack((PyFrameObject*)f); Py_DECREF(f); f = prev;
#else
    PyObject* prev = (PyObject*)((PyFrameObject*)f)->f_back; Py_XINCREF(prev); Py_XDECREF(f); f = prev;
#endif
  }
  PyGILState_Release(gs);
  return out;
}
static inline std::string capture_python_stack_safe() {
  return Py_IsInitialized() ? capture_python_stack_impl() : std::string();
}

// ========= Native sink (CUPTI-like) =========
struct NHEvent {
  const char* api;
  int device;        // optional; -1 if unknown
  uint32_t stream;   // optional; 0 if unknown
  uint64_t tsc;      // monotonic approximate timestamp
  std::vector<std::tuple<std::string,int,std::string>> frames; // root->leaf
};

static std::mutex           g_q_mu;
static std::deque<NHEvent>  g_q;
static size_t               g_q_cap = 200000;
static std::atomic<uint64_t> g_q_dropped{0};
static std::atomic<uint64_t> g_total_enqueued{0};

enum class SinkMode : int { OFF=0, FILE=1, MEM=2 };
static std::atomic<SinkMode> g_sink_mode{SinkMode::OFF};

static std::atomic<bool>    g_run_flush{false};
static std::thread          g_flush_thread;
static std::atomic<uint64_t> g_flush_req{0};

// File sink (NDJSON)
static std::mutex   g_file_mu;
static FILE*        g_ndjson = nullptr;
static std::string  g_ndjson_path;

static inline void jesc_append(std::string& o, const char* s) {
  if (!s) return;
  for (; *s; ++s) {
    switch(*s){ case '\\': o+="\\\\"; break; case '"':  o+="\\\""; break;
      case '\n': o+="\\n"; break; case '\r': o+="\\r"; break; case '\t': o+="\\t"; break;
      default: o.push_back(*s); break; }
  }
}

static void write_ndjson_line(FILE* f, const NHEvent& e) {
  std::string buf; buf.reserve(512);
  buf += "{\"op\":\""; jesc_append(buf, e.api); buf += "\"";
  if (e.device >= 0) { buf += ",\"device\":"; buf += std::to_string(e.device); }
  buf += ",\"stream\":"; buf += std::to_string((unsigned)e.stream);
  if (e.tsc) { buf += ",\"tsc\":"; buf += std::to_string(e.tsc); }
  buf += ",\"stack\":[";
  for (size_t i=0;i<e.frames.size();++i){
    const auto& fr = e.frames[i];
    buf += "[\""; jesc_append(buf, std::get<0>(fr).c_str()); buf += "\",";
    buf += std::to_string(std::get<1>(fr)); buf += ",\"";
    jesc_append(buf, std::get<2>(fr).c_str()); buf += "\"]";
    if (i+1<e.frames.size()) buf += ",";
  }
  buf += "]}\n";
  fwrite(buf.data(), 1, buf.size(), f);
}

static inline void push_event(NHEvent&& e) {
  std::lock_guard<std::mutex> lk(g_q_mu);
  if (g_q.size() >= g_q_cap) { g_q.pop_front(); g_q_dropped.fetch_add(1, std::memory_order_relaxed); }
  g_q.emplace_back(std::move(e));
  g_total_enqueued.fetch_add(1, std::memory_order_relaxed);
}

static bool open_ndjson_once(const std::string& path) {
  std::lock_guard<std::mutex> lk(g_file_mu);
  if (g_ndjson) return true;
  mkdir_p_for(path);
  g_ndjson = ::fopen(path.c_str(), "a");
  if (!g_ndjson) return false;
  g_ndjson_path = path;
  return true;
}
static void close_ndjson() {
  std::lock_guard<std::mutex> lk(g_file_mu);
  if (g_ndjson) { ::fflush(g_ndjson); ::fclose(g_ndjson); g_ndjson=nullptr; }
  g_ndjson_path.clear();
}
static void flush_ndjson_now() {
  std::lock_guard<std::mutex> lk(g_file_mu);
  if (g_ndjson) ::fflush(g_ndjson);
}

static void flusher_loop() {
  std::deque<NHEvent> local;
  while (g_run_flush.load(std::memory_order_relaxed)) {
    {
      std::lock_guard<std::mutex> lk(g_q_mu);
      if (!g_q.empty()) local.swap(g_q);
    }
    if (!local.empty()) {
      if (g_sink_mode.load() == SinkMode::FILE) {
        std::lock_guard<std::mutex> lk(g_file_mu);
        if (g_ndjson) {
          for (auto& e : local) write_ndjson_line(g_ndjson, e);
          ::fflush(g_ndjson);
        }
      }
      // MEM mode drains only via pyhook_drain()
      local.clear();
    }
    uint64_t before = g_flush_req.load(std::memory_order_relaxed);
    for (int i=0;i<10;i++){
      if (!g_run_flush.load(std::memory_order_relaxed)) break;
      if (g_flush_req.load(std::memory_order_relaxed) != before) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
  }
  // final drain
  {
    std::lock_guard<std::mutex> lk(g_q_mu);
    if (!g_q.empty()) {
      if (g_sink_mode.load() == SinkMode::FILE) {
        std::lock_guard<std::mutex> lk2(g_file_mu);
        if (g_ndjson) {
          for (auto& e : g_q) write_ndjson_line(g_ndjson, e);
          ::fflush(g_ndjson);
        }
      }
      g_q.clear();
    }
  }
}

// ========= Python-visible C API =========
extern "C" {

// Start FILE mode sink
void pyhook_start_file(const char* out_path) {
  if (!out_path || !*out_path) return;
  if (!open_ndjson_once(out_path)) return;
  g_sink_mode.store(SinkMode::FILE, std::memory_order_relaxed);
  if (!g_run_flush.exchange(true)) g_flush_thread = std::thread(flusher_loop);
}

// Start MEM mode sink (bounded queue)
void pyhook_start_mem(int limit) {
  {
    std::lock_guard<std::mutex> lk(g_q_mu);
    if (limit > 0) g_q_cap = (size_t)limit;
  }
  g_sink_mode.store(SinkMode::MEM, std::memory_order_relaxed);
  // if (!g_run_flush.exchange(true)) g_flush_thread = std::thread(flusher_loop);
}

// Drain up to N events -> Python list[dict] (schema compatible with CUPTI drain)
PyObject* pyhook_drain(int max_items) {
  if (max_items <= 0) max_items = 5000;

  // Acquire the GIL since this function builds Python objects and
  // may be called via ctypes.CDLL (GIL not held).
  PyGILState_STATE gs = PyGILState_Ensure();

  std::deque<NHEvent> local;
  {
    std::lock_guard<std::mutex> lock(g_q_mu);
    size_t n = std::min((size_t)max_items, g_q.size());
    for (size_t i = 0; i < n; ++i) {
      local.emplace_back(std::move(g_q.front()));
      g_q.pop_front();
    }
  }

  PyObject* list = PyList_New(local.size());
  size_t idx = 0;
  for (auto &e : local) {
    PyObject* py_stack = PyList_New(e.frames.size());
    for (ssize_t i = (ssize_t)e.frames.size() - 1, j = 0; i >= 0; --i, ++j) {
      const auto& fr = e.frames[(size_t)i];  // reverse: root→leaf order in py_stack
      PyObject* triple = PyList_New(3);
      PyList_SET_ITEM(triple, 0, PyUnicode_FromString(std::get<0>(fr).c_str()));
      PyList_SET_ITEM(triple, 1, PyLong_FromLong(std::get<1>(fr)));
      PyList_SET_ITEM(triple, 2, PyUnicode_FromString(std::get<2>(fr).c_str()));
      PyList_SET_ITEM(py_stack, j, triple);
    }
    PyObject* d = PyDict_New();
    PyDict_SetItemString(d, "kernel", PyUnicode_FromString(e.api ? e.api : ""));
    PyDict_SetItemString(d, "device", PyLong_FromLong(e.device >= 0 ? e.device : -1));
    PyDict_SetItemString(d, "stream", PyLong_FromUnsignedLong(e.stream));
    PyDict_SetItemString(d, "start", PyLong_FromUnsignedLongLong(e.tsc));
    PyDict_SetItemString(d, "end",   PyLong_FromUnsignedLongLong(e.tsc));
    PyDict_SetItemString(d, "stack", py_stack);
    Py_DECREF(py_stack);
    PyList_SET_ITEM(list, idx++, d);
  }

  // Let Python continue with the object safely
  PyGILState_Release(gs);
  return list;
}


// Ask flusher to flush now
void pyhook_flush() {
  g_flush_req.fetch_add(1, std::memory_order_relaxed);
  flush_ndjson_now();
}

// Stop sink, join thread, close file (if any)
void pyhook_stop() {
  if (g_run_flush.exchange(false)) {
    if (g_flush_thread.joinable()) g_flush_thread.join();
  }
  close_ndjson();
  g_sink_mode.store(SinkMode::OFF, std::memory_order_relaxed);
}

// atexit safety
void pyhook_force_dump() { pyhook_stop(); }

} // extern "C"

// ========= Unified logger (hot path) =========
static inline void log_stack_with_rf(const char* api) {
  const bool silent = env_silent();

  resolve_rf_syms_once();

  std::vector<std::tuple<std::string,int,std::string>> frames;

  const char* pstk = RF_STACK ? RF_STACK() : nullptr;

  if (pstk && *pstk) {
    auto parsed = parse_stack_text(pstk);
    frames = std::move(parsed.second);
  } else if (Py_IsInitialized()) {
    std::string text = capture_python_stack_impl();
    auto parsed = parse_stack_text(text);
    frames = std::move(parsed.second);
  }

  g_any_event.store(true, std::memory_order_relaxed);

  uint64_t c = g_evt_count.fetch_add(1, std::memory_order_relaxed) + 1;
  if (c == 1 && !silent) {
    fprintf(stderr, "[hook] first event in pid=%d tid=%ld\n", (int)self_pid(), self_tid());
  }

  // Per-site aggregation (kept cheap & lock-scoped)
  std::string site = site_from_frames(frames);
  // printf("site: %s\n", site.c_str());
  {
    std::lock_guard<std::mutex> lk(g_agg_mu);
    auto& b = g_by_site[site];
    b.count += 1;
    b.ops.insert(api ? api : "");
    std::string k = frames_key_exact(frames);
    auto it = b.seen.find(k);
    if (it == b.seen.end()) {
      size_t idx = b.callstacks.size();
      b.seen.emplace(std::move(k), idx);
      b.callstacks.push_back(StackEntry{frames});
      b.stack_counts.push_back(1);
    } else {
      b.stack_counts[it->second] += 1;
    }
  }

  // Enqueue event (CUPTI-like sink)
  NHEvent ev;
  ev.api    = api;
  ev.device = -1;      // unknown here
  ev.stream = 0;       // unknown here
  ev.tsc    = (uint64_t)std::chrono::steady_clock::now().time_since_epoch().count();
  ev.frames = std::move(frames);
  push_event(std::move(ev));

  // Optional Python aggregated snapshot cadence (count-based)
  if (int N = env_dump_every()) {
    uint64_t v = g_since_last_flush.fetch_add(1, std::memory_order_relaxed) + 1;
    if (v >= (uint64_t)N) {
      uint64_t expect = v;
      if (g_since_last_flush.compare_exchange_strong(expect, 0ull)) {
        python_sink_flush(/*final_flush=*/false);
      }
    }
  }
}

// ========= Wrapping (CUDA runtime/driver/cuBLAS) =========

// Config: library groups & search lists
static const char* kRtSos[]       = {"libcudart.so.13","libcudart.so.12","libcudart.so.11","libcudart.so",nullptr};
static const char* kDrvSo[]       = {"libcuda.so",nullptr};
static const char* kCuBlasSos[]   = {"libcublas.so.12","libcublas.so.11","libcublas.so",nullptr};
static const char* kCuBlasLtSos[] = {"libcublasLt.so.12","libcublasLt.so.11","libcublasLt.so",nullptr};

// Helpers to pick the right NOT_SUPPORTED per family
#define RUNTIME_NOSUPP      cudaErrorNotSupported
#define DRIVER_NOSUPP       CUDA_ERROR_NOT_SUPPORTED
#define CUBLAS_NOSUPP       CUBLAS_STATUS_NOT_SUPPORTED

// Declare a wrapper that resolves and logs
#define WRAP_RESOLVE(ret_not_supported, SO_GROUP, PRINT_NAME, REAL_VAR, FP_TYPE, ...) \
  static FP_TYPE REAL_VAR = nullptr;                                                   \
  if (!(REAL_VAR)) {                                                                   \
    REAL_VAR = resolve_any_of<FP_TYPE>(SO_GROUP, { __VA_ARGS__ });                     \
  }                                                                                    \
  HookScope _hs; if (_hs) log_stack_with_rf(PRINT_NAME);                               \
  if (!(REAL_VAR)) return ret_not_supported;

// --- Runtime: cudaLaunchKernel
extern "C" cudaError_t cudaLaunchKernel(const void* f, dim3 g, dim3 b,
                                        void** args, size_t shmem, cudaStream_t s) {
  using fp_t = decltype(&::cudaLaunchKernel);
  WRAP_RESOLVE(RUNTIME_NOSUPP, kRtSos, "cudaLaunchKernel", real_cudaLaunchKernel, fp_t,
               "cudaLaunchKernel");
  return real_cudaLaunchKernel(f, g, b, args, shmem, s);
}

#if CUDART_VERSION >= 12000
// Current Torch CUDA builds launch through the extensible runtime API.
extern "C" cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t* config,
                                           const void* f, void** args) {
  using fp_t = decltype(&::cudaLaunchKernelExC);
  WRAP_RESOLVE(RUNTIME_NOSUPP, kRtSos, "cudaLaunchKernelExC", real_cudaLaunchKernelExC, fp_t,
               "cudaLaunchKernelExC");
  return real_cudaLaunchKernelExC(config, f, args);
}
#endif

#if CUDART_VERSION >= 13000
// CUDA 13 compiler-generated launches use kernel handles (crt/device_functions.h).
extern "C" cudaError_t __cudaLaunchKernel(cudaKernel_t kernel, dim3 grid, dim3 block,
                                          void** args, size_t shared, cudaStream_t stream) {
  using fp_t = cudaError_t (*)(cudaKernel_t, dim3, dim3, void**, size_t, cudaStream_t);
  WRAP_RESOLVE(RUNTIME_NOSUPP, kRtSos, "__cudaLaunchKernel", real_handle_launch, fp_t,
               "__cudaLaunchKernel");
  return real_handle_launch(kernel, grid, block, args, shared, stream);
}
#endif

// --- Runtime: cudaMemcpyAsync
extern "C" cudaError_t cudaMemcpyAsync(void* d, const void* src, size_t n,
                                       cudaMemcpyKind k, cudaStream_t st) {
  using fp_t = decltype(&::cudaMemcpyAsync);
  WRAP_RESOLVE(RUNTIME_NOSUPP, kRtSos, "cudaMemcpyAsync", real_cudaMemcpyAsync, fp_t,
               "cudaMemcpyAsync");
  return real_cudaMemcpyAsync(d, src, n, k, st);
}

// --- Runtime: cudaMemsetAsync
extern "C" cudaError_t cudaMemsetAsync(void* d, int v, size_t n, cudaStream_t st) {
  using fp_t = decltype(&::cudaMemsetAsync);
  WRAP_RESOLVE(RUNTIME_NOSUPP, kRtSos, "cudaMemsetAsync", real_cudaMemsetAsync, fp_t,
               "cudaMemsetAsync");
  return real_cudaMemsetAsync(d, v, n, st);
}

// --- Driver: cuLaunchKernel
extern "C" CUresult cuLaunchKernel(CUfunction f,
                                   unsigned gx, unsigned gy, unsigned gz,
                                   unsigned bx, unsigned by, unsigned bz,
                                   unsigned shmem, CUstream s,
                                   void** params, void** extra) {
  using fp_t = decltype(&::cuLaunchKernel);
  WRAP_RESOLVE(DRIVER_NOSUPP, kDrvSo, "cuLaunchKernel", real_cuLaunchKernel, fp_t,
               "cuLaunchKernel");
  return real_cuLaunchKernel(f, gx, gy, gz, bx, by, bz, shmem, s, params, extra);
}

// --- Graph replay: cudaGraphLaunch / cuGraphLaunch
extern "C" cudaError_t cudaGraphLaunch(cudaGraphExec_t g, cudaStream_t s) {
  using fp_t = decltype(&::cudaGraphLaunch);
  WRAP_RESOLVE(RUNTIME_NOSUPP, kRtSos, "cudaGraphLaunch", real_cudaGraphLaunch, fp_t,
               "cudaGraphLaunch");
  return real_cudaGraphLaunch(g, s);
}
extern "C" CUresult cuGraphLaunch(CUgraphExec g, CUstream s) {
  using fp_t = decltype(&::cuGraphLaunch);
  WRAP_RESOLVE(DRIVER_NOSUPP, kDrvSo, "cuGraphLaunch", real_cuGraphLaunch, fp_t,
               "cuGraphLaunch");
  return real_cuGraphLaunch(g, s);
}

// --- Runtime alloc/free
extern "C" cudaError_t cudaMalloc(void** devPtr, size_t size) {
  using fp_t = decltype(&::cudaMalloc);
  WRAP_RESOLVE(RUNTIME_NOSUPP, kRtSos, "cudaMalloc", real_cudaMalloc, fp_t,
               "cudaMalloc");
  return real_cudaMalloc(devPtr, size);
}
extern "C" cudaError_t cudaFree(void* devPtr) {
  using fp_t = decltype(&::cudaFree);
  WRAP_RESOLVE(RUNTIME_NOSUPP, kRtSos, "cudaFree", real_cudaFree, fp_t,
               "cudaFree");
  return real_cudaFree(devPtr);
}

// --- cuBLAS: prefer Ex / Lt, but also catch classic s/dgemm_v2
using cublasGemmEx_fp = cublasStatus_t(*)(cublasHandle_t,
    cublasOperation_t, cublasOperation_t,
    int, int, int,
    const void*,
    const void*, cudaDataType, int,
    const void*, cudaDataType, int,
    const void*,
    void*, cudaDataType, int,
    cublasComputeType_t, cublasGemmAlgo_t);

extern "C" cublasStatus_t cublasGemmEx(cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const void* alpha,
    const void* A, cudaDataType Atype, int lda,
    const void* B, cudaDataType Btype, int ldb,
    const void* beta,
    void* C, cudaDataType Ctype, int ldc,
    cublasComputeType_t computeType,
    cublasGemmAlgo_t algo)
{
  static cublasGemmEx_fp real_cublasGemmEx = nullptr;
  if (!real_cublasGemmEx) {
    real_cublasGemmEx =
      resolve_any_of<cublasGemmEx_fp>(kCuBlasSos, {"cublasGemmEx", "cublasGemmEx_64"});
  }
  HookScope _hs; if (_hs) log_stack_with_rf("cublasGemmEx");
  if (!real_cublasGemmEx) return CUBLAS_STATUS_NOT_SUPPORTED;

  return real_cublasGemmEx(handle, transa, transb, m, n, k,
                           alpha, A, Atype, lda,
                           B, Btype, ldb, beta,
                           C, Ctype, ldc, computeType, algo);
}

extern "C" cublasStatus_t cublasSgemm(cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float* alpha,
    const float* A, int lda,
    const float* B, int ldb,
    const float* beta,
    float* C, int ldc) {
  using fp_v2 = decltype(&::cublasSgemm_v2);
  static fp_v2 real_v2 = resolve_any_of<fp_v2>(kCuBlasSos, {"cublasSgemm_v2"});
  HookScope _hs; if (_hs) log_stack_with_rf("cublasSgemm");
  if (real_v2) return real_v2(handle, transa, transb, m,n,k, alpha, A, lda, B, ldb, beta, C, ldc);
  return CUBLAS_STATUS_NOT_SUPPORTED;
}

extern "C" cublasStatus_t cublasDgemm(cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const double* alpha,
    const double* A, int lda,
    const double* B, int ldb,
    const double* beta,
    double* C, int ldc) {
  using fp_v2 = decltype(&::cublasDgemm_v2);
  static fp_v2 real_v2 = resolve_any_of<fp_v2>(kCuBlasSos, {"cublasDgemm_v2"});
  HookScope _hs; if (_hs) log_stack_with_rf("cublasDgemm");
  if (real_v2) return real_v2(handle, transa, transb, m,n,k, alpha, A, lda, B, ldb, beta, C, ldc);
  return CUBLAS_STATUS_NOT_SUPPORTED;
}

extern "C" cublasStatus_t cublasLtMatmul(
    cublasLtHandle_t            handle,
    cublasLtMatmulDesc_t        computeDesc,
    const void*                 alpha,
    const void*                 A,      cublasLtMatrixLayout_t Adesc,
    const void*                 B,      cublasLtMatrixLayout_t Bdesc,
    const void*                 beta,
    const void*                 C,      cublasLtMatrixLayout_t Cdesc,
    void*                       D,      cublasLtMatrixLayout_t Ddesc,
    const cublasLtMatmulAlgo_t* algo,
    void*                       workspace,
    size_t                      workspaceSizeInBytes,
    cudaStream_t                stream) {
  using fp_t = decltype(&::cublasLtMatmul);
  WRAP_RESOLVE(CUBLAS_NOSUPP, kCuBlasLtSos, "cublasLtMatmul", real_cublasLtMatmul, fp_t,
               "cublasLtMatmul");
  return real_cublasLtMatmul(handle, computeDesc, alpha,
                             A, Adesc, B, Bdesc, beta,
                             C, Cdesc, D, Ddesc, algo, workspace, workspaceSizeInBytes, stream);
}

// ========= atexit =========
static inline void ensure_atexit_once() {
  static std::atomic<bool> reg{false};
  bool expected = false;
  if (reg.compare_exchange_strong(expected, true)) {
    atexit(&pyhook_force_dump);          // stop flusher & close file
    atexit([](){ python_sink_flush(/*final_flush=*/true); }); // optional final aggregated snapshot
  }
}

// Force constructor to register atexit when the .so loads (optional)
__attribute__((constructor))
static void _hook_ctor() {
  ensure_atexit_once();
}


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

// --- identity helpers ---
static inline pid_t self_pid() { return getpid(); }
static inline long  self_tid() { return syscall(SYS_gettid); }

static inline std::string detect_rank() {
  const char* cands[] = {
    "RANK","OMPI_COMM_WORLD_RANK","SLURM_PROCID",
    "PMI_RANK","LOCAL_RANK", nullptr
  };
  for (const char** p=cands; *p; ++p) {
    const char* v = getenv(*p);
    if (v && *v) return std::string(v);
  }
  return "0";
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

// --- resolve the output template (PYHOOK_JSON_TPL > PYHOOK_JSON_OUT) ---
static inline const char* env_json_tpl_raw() {
  const char* t = getenv("PYHOOK_JSON_TPL");
  if (t && *t) return t;
  const char* o = getenv("PYHOOK_JSON_OUT");
  return (o && *o) ? o : nullptr;
}

// --- format template by replacing placeholders ---
static inline std::string format_path_from_tpl() {
  const char* raw = env_json_tpl_raw();
  if (!raw) return std::string(); // disabled
  std::string tpl(raw);
  auto replace_all = [&](const char* key, const std::string& val){
    std::string k = std::string("{") + key + "}";
    size_t pos = 0;
    while ((pos = tpl.find(k, pos)) != std::string::npos) {
      tpl.replace(pos, k.size(), val);
      pos += val.size();
    }
  };
  replace_all("pid",      std::to_string(self_pid()));
  replace_all("tid",      std::to_string(self_tid()));
  replace_all("rank",     detect_rank());
  replace_all("hostname", hostname_short());
  replace_all("time",     iso_time());
  return tpl;
}

// Whether template mentions {tid} (per-thread split)
static inline bool tpl_is_per_thread() {
  const char* raw = env_json_tpl_raw(); if (!raw) return false;
  return std::string(raw).find("{tid}") != std::string::npos;
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

// ========= Config: library groups & search lists =========
static const char* kRtSos[]       = {"libcudart.so.12","libcudart.so.11","libcudart.so",nullptr};
static const char* kDrvSo[]       = {"libcuda.so",nullptr};
static const char* kCuBlasSos[]   = {"libcublas.so.12","libcublas.so.11","libcublas.so",nullptr};
static const char* kCuBlasLtSos[] = {"libcublasLt.so.12","libcublasLt.so.11","libcublasLt.so",nullptr};

// ========= Env toggles =========
static inline bool env_silent() {
  static int v = []{ const char* e=getenv("PYHOOK_SILENT"); return (e && *e && strcmp(e,"0")!=0); }();
  return v;
}
static inline int env_stack_max() {
  static int v = []{ const char* e=getenv("PYHOOK_STACK_MAX"); return e? std::max(0, atoi(e)) : 0; }();
  return v;
}

// ========= Symbol resolution helpers =========
static void* dlopen_one(const char* so) {
  void* h = dlopen(so, RTLD_LAZY | RTLD_LOCAL | RTLD_NOLOAD);
  if (!h) h = dlopen(so, RTLD_LAZY | RTLD_LOCAL);
  return h;
}
static void* resolve_from_sos(const char* const* sos, const char* name) {
  // RTLD_NEXT first
  void* p = dlsym(RTLD_NEXT, name);
  if (p) return p;
  for (const char* const* s = sos; *s; ++s) {
    if (void* h = dlopen_one(*s)) { p = dlsym(h, name); if (p) return p; }
  }
  return nullptr;
}
// Try multiple candidate symbol names (e.g., v2 + legacy)
template <typename FP>
static FP resolve_any_of(const char* const* sos, std::initializer_list<const char*> names) {
  for (auto n : names) if (auto p = (FP)resolve_from_sos(sos, n)) return p;
  return nullptr;
}

// ========= TLS reentrancy guard =========
static thread_local bool g_in_hook = false;
struct HookScope {
  bool entered=false;
  HookScope() { if (!g_in_hook) { g_in_hook=true; entered=true; } }
  ~HookScope(){ if (entered) g_in_hook=false; }
  explicit operator bool() const { return entered; }
};

// ===================== Aggregation & JSON dump ====================
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
static std::atomic<bool> g_dump_registered{false};
static std::atomic<bool> g_any_event{false};   // any event seen (process-wide)

static bool env_emit_stacks_all() {
  static int v = []{ const char* e=getenv("PYHOOK_STACKS_ALL"); return (e && *e && strcmp(e,"0")!=0); }();
  return v;
}
static int env_max_callstacks_per_site() {
  static int v = []{ const char* e=getenv("PYHOOK_MAX_STACKS_PER_SITE"); return e? std::max(1, atoi(e)) : 8; }();
  return v;
}
static bool env_emit_stack_counts() {
  static int v = []{ const char* e=getenv("PYHOOK_STACK_COUNTS"); return (e && *e && strcmp(e,"0")!=0); }();
  return v;
}

// Deterministic key for a frames vector.
// Format: file ':' line ':' func '\n' …
// (If you want “fuzzy” dedup later, you can drop :line here or clamp inside normalize.)
static inline std::string frames_key(const std::vector<std::tuple<std::string,int,std::string>>& frames) {
  std::string k;
  k.reserve(frames.size() * 64);
  for (const auto& fr : frames) {
    k.append(std::get<0>(fr)); k.push_back(':');
    k.append(std::to_string(std::get<1>(fr))); k.push_back(':');
    k.append(std::get<2>(fr)); k.push_back('\n');
  }
  return k;
}

// --- add this helper near other small utils ---
static inline const char* env_stack_dir() {
  // "top" -> root→leaf (top→bottom), "bottom" -> leaf→root
  // default "top" per your request
  static const char* v = []{
    const char* e = getenv("PYHOOK_STACK_DIR");
    return (e && *e) ? e : "top";
  }();
  return v;
}

// Ensure frames are root→leaf when writing/aggregating
static inline void normalize_root_to_leaf(
    std::vector<std::tuple<std::string,int,std::string>>& frames) {
  if (!frames.empty() && strcmp(env_stack_dir(), "top") == 0) {
    // Our capture (and many RF providers) usually give leaf→root; flip to root→leaf.
    std::reverse(frames.begin(), frames.end());
  }
}


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
  // site is chosen later via site_from_frames(); return empty here for clarity
  return {std::string(), std::move(frames)};
}

// choose site key: root (default) or leaf if PYHOOK_SITE_MODE=leaf
static inline std::string site_from_frames(const std::vector<std::tuple<std::string,int,std::string>>& frames) {
  if (frames.empty()) return "(no Python frames)";
  const char* mode = getenv("PYHOOK_SITE_MODE");
  bool use_leaf = (mode && strcmp(mode, "leaf")==0);
  const auto& f = use_leaf ? frames.back() : frames.front();
  return std::get<0>(f) + std::string(":") + std::to_string(std::get<1>(f)) + " in " + std::get<2>(f);
}

static void dump_json_at_exit() {
  // Skip if we never saw an event AND the in-memory aggregate is empty.
  if (!g_any_event.load(std::memory_order_relaxed)) {
    std::lock_guard<std::mutex> lk(g_agg_mu);
    if (g_by_site.empty()) return; // nothing to dump
  }

  const char* raw = env_json_tpl_raw();
  const char* outp_env = getenv("PYHOOK_JSON_OUT"); // legacy single-file mode
  // Prefer template (single per-process file for aggregate)
  std::string outp = raw ? format_path_from_tpl() :
                     (outp_env && *outp_env ? std::string(outp_env) : std::string());

  // If still empty (streaming disabled and no aggregate desired), just skip
  if (outp.empty()) return;

  // For aggregate, write one per-process file; avoid clobber NDJSON if same
  // Use ".aggregate.json" sibling when NDJSON template equals this path
  std::string agg_path = outp;
  if (agg_path.size() >= 7 && agg_path.rfind(".ndjson") == agg_path.size()-7) {
    agg_path.replace(agg_path.size()-7, 7, ".aggregate.json");
  } else {
    agg_path += ".aggregate.json";
  }

  std::lock_guard<std::mutex> lk(g_agg_mu);
  if (g_by_site.empty()) return; // nothing aggregated

  std::ofstream f(agg_path, std::ios::out | std::ios::trunc);
  if (!f.good()) return;

  // existing aggregate JSON emission, but to f instead of std::cerr:
  std::ostream& os = f;
  os << "{\n";
  bool first_site = true;
  for (auto& kv : g_by_site) {
    if (!first_site) os << ",\n";
    first_site = false;
    const std::string& site = kv.first;
    const SiteBucket& b = kv.second;

    os << "  " << std::quoted(site) << ": {\n";
    os << "    \"count\": " << b.count << ",\n";

    os << "    \"callstacks\": [\n";
    for (size_t i = 0; i < b.callstacks.size(); ++i) {
      const auto& se = b.callstacks[i];
      os << "      [\n";
      for (size_t j = se.frames.size(); j > 0; --j) {
        const auto& fr = se.frames[j-1];
        os << "        [" << std::quoted(std::get<0>(fr)) << ", "
           << std::get<1>(fr) << ", " << std::quoted(std::get<2>(fr)) << "]";
        if (j - 1 > 0) os << ",";
        os << "\n";
      }
      os << "      ]";
      if (i + 1 < b.callstacks.size()) os << ",";
      os << "\n";
    }
    os << "    ],\n";

    os << "    \"ops\": [";
    bool first = true;
    for (const auto& op : b.ops) {
      if (!first) os << ", ";
      first = false;
      os << std::quoted(op);
    }
    os << "]\n";

    os << "  }";
  }
  os << "\n}\n";
}


// ===================== Streaming NDJSON output (optional) ====================
static inline bool env_stream_json() {
  static int v = []{ const char* e = getenv("PYHOOK_STREAM_JSON"); return (e && *e && strcmp(e,"0")!=0); }();
  return v;
}
static inline bool env_stream_dedup() {
  static int v = []{ const char* e = getenv("PYHOOK_STREAM_DEDUP"); return (e && *e && strcmp(e,"0")!=0); }();
  return v;
}
static inline bool env_emit_counts_at_exit() {
  static int v = []{ const char* e = getenv("PYHOOK_EMIT_COUNTS_AT_EXIT"); return (e && *e && strcmp(e,"0")!=0); }();
  return v;
}

static std::mutex g_json_mu;
// static std::unique_ptr<std::ofstream> g_json_os;
static std::unordered_map<std::string, uint64_t> g_stream_counts; // frames_key -> count
// One writer per thread (filename depends on template and the thread id)
static thread_local std::unique_ptr<std::ofstream> tl_json_os;
static thread_local std::string tl_json_path;

static void ensure_json_open_once() {
  if (tl_json_os) return;
  std::string path = format_path_from_tpl();
  if (path.empty()) return; // streaming disabled

  // Create/open append; best-effort (directories should exist already)
  tl_json_os.reset(new std::ofstream(path, std::ios::out | std::ios::app));
  if (tl_json_os->good()) {
    tl_json_path = std::move(path);
  } else {
    tl_json_os.reset(); // disable on failure
  }
}


// exact-key (includes line numbers as-is)
// Exact key with line numbers, **leaf -> root** order.
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


// one-line JSON writer (uses std::quoted for correct escaping)
static void ndjson_write_event(const std::string& site,
                               const char* api,
                               const std::vector<std::tuple<std::string,int,std::string>>& frames) {
  if (!env_stream_json()) return;
  ensure_json_open_once();
  if (!tl_json_os || !tl_json_os->good()) return;

  // Optional streaming dedup on exact (site,frames) scoped to process
  static std::mutex dedup_mu;
  static std::unordered_set<std::string> seen; // process-wide
  bool do_dedup = env_stream_dedup();
  std::string key;
  if (do_dedup) {
    key.reserve(site.size()+1 + frames.size()*64);
    key.append(site).push_back('|');
    key.append(frames_key_exact(frames));
    std::lock_guard<std::mutex> lk(dedup_mu);
    if (!seen.insert(key).second) {
      g_stream_counts[key] += 1;
      return;
    }
  }

  auto& os = *tl_json_os;

  // If multiple threads share one file (no {tid}), guard writes
  if (!tpl_is_per_thread()) g_json_mu.lock();

  os << "{"
     << "\"site\":" << std::quoted(site) << ","
     << "\"op\":"   << std::quoted(api ? api : "") << ","
     << "\"frames\":[";
  for (size_t i = frames.size(); i > 0; --i) {
    const auto& fr = frames[i-1];
    os << "[" << std::quoted(std::get<0>(fr)) << "," << std::get<1>(fr)
       << "," << std::quoted(std::get<2>(fr)) << "]";
    if (i > 1) os << ",";
  }
  os << "]}\n";
  os.flush();

  g_stream_counts[do_dedup ? key : frames_key_exact(frames)] += 1;

  if (!tpl_is_per_thread()) g_json_mu.unlock();
}


// optional counts summary at exit (as a single JSON object with per-key counts)

static void dump_stream_counts_at_exit() {
  if (!(env_stream_json() && env_emit_counts_at_exit())) return;
  if (!g_any_event.load(std::memory_order_relaxed)) return; // no events → skip

  // Reopen our own per-thread file? No: write counts once per-process.
  // Use the process template path, but coalesce to ".counts.json"
  std::string base = format_path_from_tpl();
  if (base.empty()) return;
  std::string path = base;
  if (path.size() >= 7 && path.rfind(".ndjson") == path.size()-7) {
    path.replace(path.size()-7, 7, ".counts.json");
  } else {
    path += ".counts.json";
  }

  std::lock_guard<std::mutex> lk(g_json_mu);
  std::ofstream os(path, std::ios::out | std::ios::trunc);
  if (!os.good()) return;

  os << "{\"event\":\"counts\",\"items\":[\n";
  bool first = true;
  for (const auto& kv : g_stream_counts) {
    if (!first) os << ",\n";
    first = false;
    os << "  {\"key\":" << std::quoted(kv.first)
       << ",\"count\":" << kv.second << "}";
  }
  os << "\n]}\n";
  os.flush();
}


static inline void ensure_dump_registered_once() {
  static std::atomic<bool> reg{false};
  bool expected = false;
  if (reg.compare_exchange_strong(expected, true)) {
    atexit(&dump_json_at_exit);           // your existing aggregate dump (optional)
    atexit(&dump_stream_counts_at_exit);  // optional counts summary for NDJSON mode
  }
}

// ========= Python stack capture (fallback) =========
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

// ========= Unified logger (prefers RF) =========
static inline void log_stack_with_rf(const char* api) {
  if (env_silent()) return;
  ensure_dump_registered_once();
  resolve_rf_syms_once();

  std::string site;
  std::vector<std::tuple<std::string,int,std::string>> frames;

  const char* pstk = RF_STACK ? RF_STACK() : nullptr;

  if (pstk && *pstk) {
    auto parsed = parse_stack_text(pstk);
    frames = std::move(parsed.second);
    // Normalize frames to root→leaf when writing/aggregating
    // normalize_root_to_leaf(frames);
    site = site_from_frames(frames);   // respects line numbers (root/leaf via env)
  } else if (Py_IsInitialized()) {
    std::string text = capture_python_stack_impl();
    auto parsed = parse_stack_text(text);
    frames = std::move(parsed.second);
    // Normalize frames to root→leaf when writing/aggregating
    // normalize_root_to_leaf(frames);
    site = site_from_frames(frames);
  } else {
    site = "(no Python frames)";
  }
  // inside log_stack_with_rf(...) just before/after ndjson_write_event(...)
  g_any_event.store(true, std::memory_order_relaxed);


  // ---- write one JSON line immediately (exact line numbers preserved)
  ndjson_write_event(site, api, frames);

  // (Optional) keep your existing aggregated map + dump_json_at_exit() if you still want it.
  std::lock_guard<std::mutex> lk(g_agg_mu);
  auto& b = g_by_site[site];
  b.count += 1;
  b.ops.insert(api);
  // unique-per-site (exact) dedup in memory so aggregated JSON still compact:
  std::string k = frames_key_exact(frames);
  auto it = b.seen.find(k);
  if (it == b.seen.end()) {
    size_t idx = b.callstacks.size();
    b.seen.emplace(std::move(k), idx);
    b.callstacks.push_back(StackEntry{std::move(frames)});
    b.stack_counts.push_back(1);
  } else {
    b.stack_counts[it->second] += 1;
  }
}




// ========= Minimal macros to eliminate redundancy =========
// Declare a wrapper that:
//   - resolves any of {SYMS...} from the given SO group,
//   - logs (with RF), reentrancy-guards, null-guards,
//   - returns NOT_SUPPORTED if missing.
#define WRAP_RESOLVE(ret_not_supported, SO_GROUP, PRINT_NAME, REAL_VAR, FP_TYPE, ...) \
  static FP_TYPE REAL_VAR = nullptr;                                                   \
  if (!(REAL_VAR)) {                                                                   \
    REAL_VAR = resolve_any_of<FP_TYPE>(SO_GROUP, { __VA_ARGS__ });                     \
    /* REAL_VAR may remain null if symbol not present; we handle below */              \
  }                                                                                    \
  HookScope _hs; if (_hs) log_stack_with_rf(PRINT_NAME);                               \
  if (!(REAL_VAR)) return ret_not_supported;

// Helpers to pick the right NOT_SUPPORTED per family
#define RUNTIME_NOSUPP      cudaErrorNotSupported
#define DRIVER_NOSUPP       CUDA_ERROR_NOT_SUPPORTED
#define CUBLAS_NOSUPP       CUBLAS_STATUS_NOT_SUPPORTED

// ========= Wrappers (concise, version-proof) =========

// --- Runtime: cudaLaunchKernel
extern "C" cudaError_t cudaLaunchKernel(const void* f, dim3 g, dim3 b,
                                        void** args, size_t shmem, cudaStream_t s) {
  using fp_t = decltype(&::cudaLaunchKernel);
  WRAP_RESOLVE(RUNTIME_NOSUPP, kRtSos, "cudaLaunchKernel", real_cudaLaunchKernel, fp_t,
               "cudaLaunchKernel");
  return real_cudaLaunchKernel(f, g, b, args, shmem, s);
}

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
  std::cerr << "[hook] cuLaunchKernel [tid=" << syscall(SYS_gettid) << "]\n";
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

// --- Runtime alloc/free (examples; add more with same pattern)
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

// --- cuBLAS: GemmEx (explicit FP type to avoid overload/alias issues)
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
  // Try both exported names (varies by CUDA/cuBLAS build)
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

// ---- CUDA 12 launcher: cudaLaunchKernelExC (+ pt variants) ----
extern "C" cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t* cfg,
                                           const void* f, void** args) {
  using fp_t = cudaError_t (*)(const cudaLaunchConfig_t*, const void*, void**);
  WRAP_RESOLVE(RUNTIME_NOSUPP, kRtSos, "cudaLaunchKernelExC", real_cudaLaunchKernelExC, fp_t,
               "cudaLaunchKernelExC", "cudaLaunchKernelExC_ptsz", "cudaLaunchKernelExC_ptds");
  return real_cudaLaunchKernelExC(cfg, f, args);
}

extern "C" cudaError_t cudaLaunchKernelExC_ptsz(const cudaLaunchConfig_t* cfg,
                                                const void* f, void** args) {
  using fp_t = cudaError_t (*)(const cudaLaunchConfig_t*, const void*, void**);
  WRAP_RESOLVE(RUNTIME_NOSUPP, kRtSos, "cudaLaunchKernelExC_ptsz", real_cudaLaunchKernelExC_ptsz, fp_t,
               "cudaLaunchKernelExC_ptsz", "cudaLaunchKernelExC", "cudaLaunchKernelExC_ptds");
  return real_cudaLaunchKernelExC_ptsz(cfg, f, args);
}

extern "C" cudaError_t cudaLaunchKernelExC_ptds(const cudaLaunchConfig_t* cfg,
                                                const void* f, void** args) {
  using fp_t = cudaError_t (*)(const cudaLaunchConfig_t*, const void*, void**);
  WRAP_RESOLVE(RUNTIME_NOSUPP, kRtSos, "cudaLaunchKernelExC_ptds", real_cudaLaunchKernelExC_ptds, fp_t,
               "cudaLaunchKernelExC_ptds", "cudaLaunchKernelExC", "cudaLaunchKernelExC_ptsz");
  return real_cudaLaunchKernelExC_ptds(cfg, f, args);
}

// ---- cudaLaunchKernel + pt variants ----
extern "C" cudaError_t cudaLaunchKernel_ptsz(const void* f, dim3 g, dim3 b,
                                             void** args, size_t shmem, cudaStream_t s) {
  using fp_t = decltype(&::cudaLaunchKernel_ptsz);
  WRAP_RESOLVE(RUNTIME_NOSUPP, kRtSos, "cudaLaunchKernel_ptsz", real_cudaLaunchKernel_ptsz, fp_t,
               "cudaLaunchKernel_ptsz", "cudaLaunchKernel_ptds", "cudaLaunchKernel");
  return real_cudaLaunchKernel_ptsz(f, g, b, args, shmem, s);
}
extern "C" cudaError_t cudaLaunchKernel_ptds(const void* f, dim3 g, dim3 b,
                                             void** args, size_t shmem, cudaStream_t s) {
  using fp_t = decltype(&::cudaLaunchKernel_ptds);
  WRAP_RESOLVE(RUNTIME_NOSUPP, kRtSos, "cudaLaunchKernel_ptds", real_cudaLaunchKernel_ptds, fp_t,
               "cudaLaunchKernel_ptds", "cudaLaunchKernel_ptsz", "cudaLaunchKernel");
  return real_cudaLaunchKernel_ptds(f, g, b, args, shmem, s);
}

// ---- cudaMemcpyAsync + pt variants ----
extern "C" cudaError_t cudaMemcpyAsync_ptsz(void* d, const void* src, size_t n,
                                            cudaMemcpyKind k, cudaStream_t st) {
  using fp_t = decltype(&::cudaMemcpyAsync_ptsz);
  WRAP_RESOLVE(RUNTIME_NOSUPP, kRtSos, "cudaMemcpyAsync_ptsz", real_cudaMemcpyAsync_ptsz, fp_t,
               "cudaMemcpyAsync_ptsz", "cudaMemcpyAsync_ptds", "cudaMemcpyAsync");
  return real_cudaMemcpyAsync_ptsz(d, src, n, k, st);
}
extern "C" cudaError_t cudaMemcpyAsync_ptds(void* d, const void* src, size_t n,
                                            cudaMemcpyKind k, cudaStream_t st) {
  using fp_t = decltype(&::cudaMemcpyAsync_ptds);
  WRAP_RESOLVE(RUNTIME_NOSUPP, kRtSos, "cudaMemcpyAsync_ptds", real_cudaMemcpyAsync_ptds, fp_t,
               "cudaMemcpyAsync_ptds", "cudaMemcpyAsync_ptsz", "cudaMemcpyAsync");
  return real_cudaMemcpyAsync_ptds(d, src, n, k, st);
}

// ---- cudaMemsetAsync + pt variants ----
extern "C" cudaError_t cudaMemsetAsync_ptsz(void* d, int v, size_t n, cudaStream_t st) {
  using fp_t = decltype(&::cudaMemsetAsync_ptsz);
  WRAP_RESOLVE(RUNTIME_NOSUPP, kRtSos, "cudaMemsetAsync_ptsz", real_cudaMemsetAsync_ptsz, fp_t,
               "cudaMemsetAsync_ptsz", "cudaMemsetAsync_ptds", "cudaMemsetAsync");
  return real_cudaMemsetAsync_ptsz(d, v, n, st);
}
extern "C" cudaError_t cudaMemsetAsync_ptds(void* d, int v, size_t n, cudaStream_t st) {
  using fp_t = decltype(&::cudaMemsetAsync_ptds);
  WRAP_RESOLVE(RUNTIME_NOSUPP, kRtSos, "cudaMemsetAsync_ptds", real_cudaMemsetAsync_ptds, fp_t,
               "cudaMemsetAsync_ptds", "cudaMemsetAsync_ptsz", "cudaMemsetAsync");
  return real_cudaMemsetAsync_ptds(d, v, n, st);
}



extern "C" cublasStatus_t cublasSgemm(cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float* alpha,
    const float* A, int lda,
    const float* B, int ldb,
    const float* beta,
    float* C, int ldc) {
  // Resolve v2 first, then legacy (avoid XERBLA / ABI mismatch)
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

extern "C" __attribute__((visibility("default")))
void pyhook_force_dump(void) {
  // best-effort: ensure we write both artifacts even if dlclose happens
  dump_stream_counts_at_exit();
  dump_json_at_exit();
}

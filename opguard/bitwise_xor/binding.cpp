// binding.cpp
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>   // <-- needed for c10::cuda::CUDAGuard
#include <cuda_runtime_api.h>

#include <unordered_map>
#include <mutex>
#include <cstdint>
#include <algorithm>

// C-linkage wrappers from the .cu file
extern "C" void xor_first_pass_auto_or_small(const uint8_t*,  uint32_t*, uint32_t*, size_t, int, int, cudaStream_t);
extern "C" void xor_words_auto_or_small     (const uint32_t*, uint32_t*, uint32_t*, size_t, int, int, cudaStream_t);
extern "C" void xor_regional_launch(const uint8_t*, uint32_t*, uint32_t*, size_t, int, int, cudaStream_t);

// constants local to binding
static constexpr int BLOCK_SIZE = 256;
static constexpr int MAX_BLOCKS = 1024;

static inline int compute_blocks_from_words(size_t Nwords) {
    size_t nb = (Nwords + BLOCK_SIZE - 1) / (size_t)BLOCK_SIZE;
    if (nb < 1) nb = 1;
    if (nb > MAX_BLOCKS) nb = MAX_BLOCKS;
    return static_cast<int>(nb);
}

// ---------- lock-lean workspace cache ----------
struct Key { int device; uint64_t stream_key; };
struct KeyHash {
    size_t operator()(const Key& k) const noexcept {
        return (static_cast<size_t>(k.device) << 32) ^ static_cast<size_t>(k.stream_key);
    }
};
struct KeyEq {
    bool operator()(const Key& a, const Key& b) const noexcept {
        return a.device == b.device && a.stream_key == b.stream_key;
    }
};

static thread_local Key        tls_last_key{-1, 0};
static thread_local at::Tensor tls_last_ws;
static thread_local std::unordered_map<Key, at::Tensor, KeyHash, KeyEq> tls_ws;

static std::unordered_map<Key, at::Tensor, KeyHash, KeyEq> g_ws_shared;
static std::mutex g_ws_mu;

static at::Tensor get_workspace_for(const c10::Device& dev, cudaStream_t stream) {
    const uint64_t sk = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(stream));
    Key k{dev.index(), sk};

    if (tls_last_ws.defined() && tls_last_key.device == k.device && tls_last_key.stream_key == k.stream_key) {
        return tls_last_ws;
    }

    auto it = tls_ws.find(k);
    if (it != tls_ws.end()) {
        tls_last_key = k; tls_last_ws = it->second;
        return it->second;
    }

    {
        std::lock_guard<std::mutex> guard(g_ws_mu);
        auto jt = g_ws_shared.find(k);
        if (jt != g_ws_shared.end()) {
            tls_ws.emplace(k, jt->second);
            tls_last_key = k; tls_last_ws = jt->second;
            return jt->second;
        }
    }

    auto opts = at::TensorOptions().device(dev).dtype(at::kInt);
    at::Tensor t = at::empty({MAX_BLOCKS}, opts);

    tls_ws.emplace(k, t);
    {
        std::lock_guard<std::mutex> guard(g_ws_mu);
        g_ws_shared.emplace(k, t);
    }
    tls_last_key = k; tls_last_ws = t;
    return t;
}

static void clear_workspace_cache() {
    tls_last_ws.reset();
    tls_ws.clear();
    std::lock_guard<std::mutex> guard(g_ws_mu);
    g_ws_shared.clear();
}

// ---------- public APIs: launch on current callsite stream ----------
at::Tensor xor_reduce_cuda(at::Tensor byte_u8) {
    TORCH_CHECK(byte_u8.is_cuda(), "input must be CUDA");
    TORCH_CHECK(byte_u8.dtype() == at::kByte, "input must be torch.uint8");
    TORCH_CHECK(byte_u8.is_contiguous(), "input must be contiguous");

    auto dev = byte_u8.device();
    c10::cuda::CUDAGuard guard(dev);
    auto s = at::cuda::getCurrentCUDAStream(dev.index());
    cudaStream_t stream = s.stream();

    const size_t Nbytes = static_cast<size_t>(byte_u8.numel());
    const size_t Nwords = (Nbytes + 3) / 4;
    const int blocks = std::max(1, compute_blocks_from_words(Nwords));

    auto out = at::empty({1}, byte_u8.options().dtype(at::kInt));

    if (Nbytes == 0) {
        cudaMemsetAsync(out.data_ptr<int32_t>(), 0, sizeof(int32_t), stream);
        return out;
    }

    at::Tensor ws_max = get_workspace_for(dev, stream);
    at::Tensor partials = ws_max.narrow(0, 0, blocks);

    xor_first_pass_auto_or_small(
        byte_u8.data_ptr<uint8_t>(),
        reinterpret_cast<uint32_t*>(partials.data_ptr<int32_t>()),
        reinterpret_cast<uint32_t*>(out.data_ptr<int32_t>()),
        Nbytes, blocks, BLOCK_SIZE, stream);

    return out;
}

at::Tensor xor_reduce_words_cuda(at::Tensor i32_words) {
    TORCH_CHECK(i32_words.is_cuda(), "u32_words must be CUDA");
    TORCH_CHECK(i32_words.dtype() == at::kInt, "expect torch.int32");
    TORCH_CHECK(i32_words.is_contiguous(), "must be contiguous");

    auto dev = i32_words.device();
    c10::cuda::CUDAGuard guard(dev);
    auto s = at::cuda::getCurrentCUDAStream(dev.index());
    cudaStream_t stream = s.stream();

    const size_t Nwords = static_cast<size_t>(i32_words.numel());
    const int blocks = std::max(1, compute_blocks_from_words(Nwords));

    auto out = at::empty({1}, i32_words.options().dtype(at::kInt));

    if (Nwords == 0) {
        cudaMemsetAsync(out.data_ptr<int32_t>(), 0, sizeof(int32_t), stream);
        return out;
    }

    at::Tensor ws_max = get_workspace_for(dev, stream);
    at::Tensor partials = ws_max.narrow(0, 0, blocks);

    xor_words_auto_or_small(
        reinterpret_cast<const uint32_t*>(i32_words.data_ptr<int32_t>()),
        reinterpret_cast<uint32_t*>(partials.data_ptr<int32_t>()),
        reinterpret_cast<uint32_t*>(out.data_ptr<int32_t>()),
        Nwords, blocks, BLOCK_SIZE, stream);

    return out;
}

at::Tensor xor_reduce_regional_cuda(at::Tensor bytes, int64_t regions) {
    TORCH_CHECK(bytes.is_cuda() && bytes.scalar_type() == at::kByte && bytes.is_contiguous(),
                "input must be contiguous CUDA uint8");
    TORCH_CHECK(regions >= 1 && regions <= 64, "regions must be in [1, 64]");
    if (regions == 1) return xor_reduce_cuda(bytes);
    const auto dev = bytes.device();
    c10::cuda::CUDAGuard guard(dev);
    const auto stream = at::cuda::getCurrentCUDAStream(dev.index()).stream();
    const size_t nbytes = bytes.numel();
    const size_t words_per_region = ((nbytes + 3) / 4 + regions - 1) / regions;
    // Retain the total workspace/block budget of the one-region path.
    const int blocks = (nbytes + 3) / 4 <= 4096 ? 1 :
        std::min(MAX_BLOCKS / static_cast<int>(regions), compute_blocks_from_words(words_per_region));
    auto out = at::empty({regions}, bytes.options().dtype(at::kInt));
    auto workspace = get_workspace_for(dev, stream);
    xor_regional_launch(bytes.data_ptr<uint8_t>(),
                        reinterpret_cast<uint32_t*>(workspace.data_ptr<int32_t>()),
                        reinterpret_cast<uint32_t*>(out.data_ptr<int32_t>()),
                        nbytes, regions, blocks, stream);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("xor_reduce_regional_cuda", &xor_reduce_regional_cuda,
          "Fixed flat-word regional XOR on current stream (no host synchronization)");
    m.def("xor_reduce_cuda",       &xor_reduce_cuda,       "XOR reduce (bytes, cached workspace) on current stream");
    m.def("xor_reduce_words_cuda", &xor_reduce_words_cuda, "XOR reduce (u32, cached workspace) on current stream");
    m.def("clear_xor_workspace_cache", &clear_workspace_cache, "Clear cached workspaces");
}

// bitwise_xor_reduce.cu
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <algorithm>

static constexpr int BLOCK_SIZE      = 256;
static constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / 32;
// Tune: one-kernel path threshold (in u32 words)
static constexpr int SMALL_WORDS_THRESH = 4096;  // ~16KB

// ======================= Common helpers =======================
__device__ __forceinline__ uint32_t warp_xor(uint32_t v) {
    const unsigned FULL = 0xFFFFFFFFu;
    #pragma unroll
    for (int ofs = 16; ofs > 0; ofs >>= 1) {
        v ^= __shfl_down_sync(FULL, v, ofs);
    }
    return v;
}

__global__ void xor_reduce_second_pass(const uint32_t* __restrict__ partial,
                                       uint32_t*       __restrict__ out,
                                       int                           numPartials)
{
    uint32_t acc = 0u;
    for (int i = threadIdx.x; i < numPartials; i += blockDim.x) {
        acc ^= partial[i];
    }
    acc = warp_xor(acc);

    __shared__ uint32_t warp_sums[WARPS_PER_BLOCK];
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;
    if (lane == 0) warp_sums[wid] = acc;
    __syncthreads();

    if (wid == 0) {
        uint32_t block_acc = (lane < WARPS_PER_BLOCK) ? warp_sums[lane] : 0u;
        block_acc = warp_xor(block_acc);
        if (lane == 0) *out = block_acc;
    }
}

// ======================= FAST PATH (u32 words) =======================
__global__ void xor_reduce_first_pass_u32(const uint32_t* __restrict__ data,
                                          uint32_t*       __restrict__ partial,
                                          size_t                        N) {
    uint32_t acc  = 0u;
    size_t idx    = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    // unroll by 4 for bandwidth
    for (; idx + 3*stride < N; idx += 4*stride) {
        acc ^= data[idx];
        acc ^= data[idx +   stride];
        acc ^= data[idx + 2*stride];
        acc ^= data[idx + 3*stride];
    }
    for (; idx < N; idx += stride) {
        acc ^= data[idx];
    }

    acc = warp_xor(acc);

    __shared__ uint32_t warp_sums[WARPS_PER_BLOCK];
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;
    if (lane == 0) warp_sums[wid] = acc;
    __syncthreads();

    if (wid == 0) {
        uint32_t block_acc = (lane < WARPS_PER_BLOCK) ? warp_sums[lane] : 0u;
        block_acc = warp_xor(block_acc);
        if (lane == 0) partial[blockIdx.x] = block_acc;
    }
}

// ======================= BYTE PATH (layout-independent) =======================
__device__ __forceinline__ uint32_t pack4_le(const uint8_t* in, size_t N, size_t b) {
    uint32_t w = 0u;
    if (b + 0 < N) w |= ((uint32_t)in[b + 0]) << 0;
    if (b + 1 < N) w |= ((uint32_t)in[b + 1]) << 8;
    if (b + 2 < N) w |= ((uint32_t)in[b + 2]) << 16;
    if (b + 3 < N) w |= ((uint32_t)in[b + 3]) << 24;
    return w;
}

__global__ void xor_reduce_first_pass_bytes(const uint8_t* __restrict__ bytes,
                                            size_t                       Nbytes,
                                            uint32_t*     __restrict__   partial) {
    const size_t Nwords = (Nbytes + 3) / 4;
    uint32_t acc  = 0u;
    size_t wid    = blockIdx.x * blockDim.x + threadIdx.x;  // word index
    size_t stride = blockDim.x * gridDim.x;

    // unroll by 4 words
    for (; wid + 3*stride < Nwords; wid += 4*stride) {
        acc ^= pack4_le(bytes, Nbytes, (wid              ) * 4);
        acc ^= pack4_le(bytes, Nbytes, (wid +   stride  ) * 4);
        acc ^= pack4_le(bytes, Nbytes, (wid + 2*stride  ) * 4);
        acc ^= pack4_le(bytes, Nbytes, (wid + 3*stride  ) * 4);
    }
    for (; wid < Nwords; wid += stride) {
        acc ^= pack4_le(bytes, Nbytes, wid * 4);
    }

    acc = warp_xor(acc);

    __shared__ uint32_t warp_sums[WARPS_PER_BLOCK];
    const int lane = threadIdx.x & 31;
    const int w    = threadIdx.x >> 5;
    if (lane == 0) warp_sums[w] = acc;
    __syncthreads();

    if (w == 0) {
        uint32_t block_acc = (lane < WARPS_PER_BLOCK) ? warp_sums[lane] : 0u;
        block_acc = warp_xor(block_acc);
        if (lane == 0) partial[blockIdx.x] = block_acc;
    }
}

// ======================= SMALL-N SINGLE-KERNEL PATHS =======================
__global__ void xor_reduce_small_u32(const uint32_t* __restrict__ data,
                                     uint32_t*       __restrict__ out,
                                     size_t                        N) {
    uint32_t acc = 0u;
    for (size_t i = threadIdx.x; i < N; i += blockDim.x) {
        acc ^= data[i];
    }
    acc = warp_xor(acc);

    __shared__ uint32_t warp_sums[WARPS_PER_BLOCK];
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;
    if (lane == 0) warp_sums[wid] = acc;
    __syncthreads();

    if (wid == 0) {
        uint32_t v = (lane < WARPS_PER_BLOCK) ? warp_sums[lane] : 0u;
        v = warp_xor(v);
        if (lane == 0) *out = v;
    }
}

__global__ void xor_reduce_small_bytes(const uint8_t* __restrict__ bytes,
                                       size_t                       Nbytes,
                                       uint32_t*     __restrict__   out) {
    const size_t N = (Nbytes + 3) / 4;  // packed words
    uint32_t acc = 0u;
    for (size_t w = threadIdx.x; w < N; w += blockDim.x) {
        const size_t b = w * 4;
        uint32_t v = 0u;
        if (b + 0 < Nbytes) v |= ((uint32_t)bytes[b + 0]) << 0;
        if (b + 1 < Nbytes) v |= ((uint32_t)bytes[b + 1]) << 8;
        if (b + 2 < Nbytes) v |= ((uint32_t)bytes[b + 2]) << 16;
        if (b + 3 < Nbytes) v |= ((uint32_t)bytes[b + 3]) << 24;
        acc ^= v;
    }
    acc = warp_xor(acc);

    __shared__ uint32_t warp_sums[WARPS_PER_BLOCK];
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;
    if (lane == 0) warp_sums[wid] = acc;
    __syncthreads();

    if (wid == 0) {
        uint32_t v = (lane < WARPS_PER_BLOCK) ? warp_sums[lane] : 0u;
        v = warp_xor(v);
        if (lane == 0) *out = v;
    }
}

// All regions share a single launch. Each word is read once, independently of
// the launch geometry. Region boundaries match flat-u32-le-v1 on the CPU.
template<bool Aligned>
__device__ __forceinline__ uint32_t regional_word(const uint8_t* bytes, size_t nbytes, size_t word) {
    if constexpr (Aligned) return reinterpret_cast<const uint32_t*>(bytes)[word];
    else return pack4_le(bytes, nbytes, 4 * word);
}

// One warp per region avoids block-wide synchronization for small tensors.
template<bool Aligned>
__global__ void xor_four_small(const uint8_t* bytes, uint32_t* out, size_t nbytes) {
    const size_t words = (nbytes + 3) / 4, base = words / 4, rem = words % 4;
    const int r = threadIdx.x / 32, local = threadIdx.x % 32;
    const size_t begin = r * base + (r < rem ? r : rem), count = base + (r < rem);
    uint32_t acc = 0;
    for (size_t j = local; j < count; j += 32)
        acc ^= regional_word<Aligned>(bytes, nbytes, begin + j);
    acc = warp_xor(acc);
    if (local == 0) out[r] = acc;
}

template<bool Aligned, int FixedRegions = 0>
__global__ void xor_regional_first(const uint8_t* bytes, uint32_t* partial,
                                   uint32_t* out, size_t nbytes, int regions) {
    const size_t words = (nbytes + 3) / 4;
    const int count_regions = FixedRegions ? FixedRegions : regions;
    const size_t base = words / count_regions, rem = words % count_regions;
    const size_t r = blockIdx.y;
    const size_t begin = r * base + (r < rem ? r : rem);
    const size_t count = base + (r < rem);
    uint32_t acc = 0;
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;
    for (; j + 3 * stride < count; j += 4 * stride) {
        acc ^= regional_word<Aligned>(bytes, nbytes, begin + j);
        acc ^= regional_word<Aligned>(bytes, nbytes, begin + j + stride);
        acc ^= regional_word<Aligned>(bytes, nbytes, begin + j + 2 * stride);
        acc ^= regional_word<Aligned>(bytes, nbytes, begin + j + 3 * stride);
    }
    for (; j < count; j += stride)
        acc ^= regional_word<Aligned>(bytes, nbytes, begin + j);
    acc = warp_xor(acc);
    __shared__ uint32_t warps[WARPS_PER_BLOCK];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) warps[wid] = acc;
    __syncthreads();
    if (wid == 0) {
        acc = warp_xor(lane < WARPS_PER_BLOCK ? warps[lane] : 0u);
        if (lane == 0) {
            if (gridDim.x == 1) out[r] = acc;
            else partial[r * gridDim.x + blockIdx.x] = acc;
        }
    }
}

__global__ void xor_regional_finish(const uint32_t* partial, uint32_t* out, int blocks) {
    uint32_t acc = 0;
    for (int j = threadIdx.x; j < blocks; j += blockDim.x)
        acc ^= partial[blockIdx.x * blocks + j];
    acc = warp_xor(acc);
    __shared__ uint32_t warps[WARPS_PER_BLOCK];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) warps[wid] = acc;
    __syncthreads();
    if (wid == 0) {
        acc = warp_xor(lane < WARPS_PER_BLOCK ? warps[lane] : 0u);
        if (lane == 0) out[blockIdx.x] = acc;
    }
}

extern "C" void xor_regional_launch(const uint8_t* bytes, uint32_t* partial,
                                     uint32_t* out, size_t nbytes, int regions,
                                     int blocks, cudaStream_t stream) {
    const bool aligned = reinterpret_cast<uintptr_t>(bytes) % 4 == 0 && nbytes % 4 == 0;
    if (regions == 4 && (nbytes + 3) / 4 <= SMALL_WORDS_THRESH) {
        if (aligned) xor_four_small<true><<<1, 128, 0, stream>>>(bytes, out, nbytes);
        else xor_four_small<false><<<1, 128, 0, stream>>>(bytes, out, nbytes);
        return;
    }
    if (regions == 4 && aligned) xor_regional_first<true, 4><<<dim3(blocks, regions), BLOCK_SIZE, 0, stream>>>(
        bytes, partial, out, nbytes, regions);
    else if (regions == 4) xor_regional_first<false, 4><<<dim3(blocks, regions), BLOCK_SIZE, 0, stream>>>(
        bytes, partial, out, nbytes, regions);
    else if (aligned) xor_regional_first<true><<<dim3(blocks, regions), BLOCK_SIZE, 0, stream>>>(
        bytes, partial, out, nbytes, regions);
    else xor_regional_first<false><<<dim3(blocks, regions), BLOCK_SIZE, 0, stream>>>(
        bytes, partial, out, nbytes, regions);
    if (blocks > 1)
        xor_regional_finish<<<regions, BLOCK_SIZE, 0, stream>>>(partial, out, blocks);
}

// ======================= C-linkage launchers (bind to provided stream) =======================

extern "C"
void xor_first_pass_u32_launch(const uint32_t* d_data,
                               uint32_t*       d_partials,
                               size_t          Nwords,
                               int             blocks,
                               int             block_size,
                               cudaStream_t    stream) {
    if (Nwords == 0) return;
    if (blocks <= 0) blocks = 1;
    if (block_size <= 0) block_size = BLOCK_SIZE;
    xor_reduce_first_pass_u32<<<blocks, block_size, 0, stream>>>(d_data, d_partials, Nwords);
}

extern "C"
void xor_first_pass_bytes_launch(const uint8_t* d_bytes,
                                 uint32_t*      d_partials,
                                 size_t         Nbytes,
                                 int            blocks,
                                 int            block_size,
                                 cudaStream_t   stream) {
    if (Nbytes == 0) return;
    if (blocks <= 0) blocks = 1;
    if (block_size <= 0) block_size = BLOCK_SIZE;
    xor_reduce_first_pass_bytes<<<blocks, block_size, 0, stream>>>(d_bytes, Nbytes, d_partials);
}

extern "C"
void xor_second_pass_launch(const uint32_t* d_partials,
                            uint32_t*       d_out,
                            int             num_partials,
                            int             block_size,
                            cudaStream_t    stream) {
    if (num_partials <= 0) {
        cudaMemsetAsync(d_out, 0, sizeof(uint32_t), stream);
        return;
    }
    if (block_size <= 0) block_size = BLOCK_SIZE;
    xor_reduce_second_pass<<<1, block_size, 0, stream>>>(d_partials, d_out, num_partials);
}

// ===== Auto-dispatch with small-path for BYTES input =====
extern "C"
void xor_first_pass_auto_or_small(const uint8_t* d_bytes,
                                  uint32_t*      d_partials,
                                  uint32_t*      d_out,
                                  size_t         Nbytes,
                                  int            blocks,
                                  int            block_size,
                                  cudaStream_t   stream) {
    if (Nbytes == 0) {
        cudaMemsetAsync(d_out, 0, sizeof(uint32_t), stream);
        return;
    }
    const uintptr_t addr = reinterpret_cast<uintptr_t>(d_bytes);
    const bool aligned = ((addr % 4) == 0) && ((Nbytes % 4) == 0);

    if (aligned) {
        const size_t Nwords = Nbytes / 4;
        if (Nwords <= SMALL_WORDS_THRESH) {
            xor_reduce_small_u32<<<1, BLOCK_SIZE, 0, stream>>>(
                reinterpret_cast<const uint32_t*>(d_bytes), d_out, Nwords);
        } else {
            xor_first_pass_u32_launch(reinterpret_cast<const uint32_t*>(d_bytes),
                                      d_partials, Nwords, blocks, block_size, stream);
            xor_second_pass_launch(d_partials, d_out, blocks, block_size, stream);
        }
    } else {
        const size_t Nwords = (Nbytes + 3) / 4;
        if (Nwords <= SMALL_WORDS_THRESH) {
            xor_reduce_small_bytes<<<1, BLOCK_SIZE, 0, stream>>>(d_bytes, Nbytes, d_out);
        } else {
            xor_first_pass_bytes_launch(d_bytes, d_partials, Nbytes, blocks, block_size, stream);
            xor_second_pass_launch(d_partials, d_out, blocks, block_size, stream);
        }
    }
}

// ===== Words input variant (u32 already) with small-path =====
extern "C"
void xor_words_auto_or_small(const uint32_t* d_words,
                             uint32_t*       d_partials,
                             uint32_t*       d_out,
                             size_t          Nwords,
                             int             blocks,
                             int             block_size,
                             cudaStream_t    stream) {
    if (Nwords == 0) {
        cudaMemsetAsync(d_out, 0, sizeof(uint32_t), stream);
        return;
    }
    if (Nwords <= SMALL_WORDS_THRESH) {
        xor_reduce_small_u32<<<1, BLOCK_SIZE, 0, stream>>>(d_words, d_out, Nwords);
    } else {
        xor_first_pass_u32_launch(d_words, d_partials, Nwords, blocks, block_size, stream);
        xor_second_pass_launch(d_partials, d_out, blocks, block_size, stream);
    }
}

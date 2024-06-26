#include "scanscam_cpu.hpp"
#include <torch/extension.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT_CUDA(x) \
    CHECK_CUDA(x);          \
    CHECK_CONTIGUOUS(x);

__device__ uint shifted_shared_index(uint idx)
{
    return idx + idx / 32;
}

template <typename T, typename T1>
__device__ __host__ T ceil_div(T num, T1 denom)
{
    return num / denom + (num % denom == 0 ? 0 : 1);
}

template <typename scalar_t>
__global__ void __launch_bounds__(64, 1) simple_linear_scan_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2> gate,
    const torch::PackedTensorAccessor32<scalar_t, 2> value,
    torch::PackedTensorAccessor32<scalar_t, 2> output)
{
    const uint outerIndex = blockIdx.x * 64 + threadIdx.x;
    if (outerIndex < gate.size(0)) {
        scalar_t acc = 0.0;
        for (uint i = 0; i < gate.size(1); i++) {
            acc *= gate[outerIndex][i];
            acc += value[outerIndex][i];
            output[outerIndex][i] = acc;
        }
    }
}

void simple_linear_scan(
    torch::Tensor gate,
    torch::Tensor value,
    torch::Tensor output)
{
    CHECK_INPUT_CUDA(gate);
    CHECK_INPUT_CUDA(value);
    CHECK_INPUT_CUDA(output);
    assert(gate.sizes() == value.sizes());
    assert(gate.sizes() == output.sizes());

    const dim3 threads(64, 1, 1);
    const dim3 blocks(ceil_div(gate.size(0), 64), 1, 1);

    AT_DISPATCH_FLOATING_TYPES(
        gate.scalar_type(),
        "simple_linear_scan",
        ([&] { simple_linear_scan_kernel<scalar_t><<<blocks, threads>>>(
                   gate.packed_accessor32<scalar_t, 2>(),
                   value.packed_accessor32<scalar_t, 2>(),
                   output.packed_accessor32<scalar_t, 2>()); }));
}

template <typename scalar_t>
__global__ void __launch_bounds__(64, 2) coalesced_linear_scan_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2> gate,
    const torch::PackedTensorAccessor32<scalar_t, 2> value,
    torch::PackedTensorAccessor32<scalar_t, 2> output)
{
    __shared__ scalar_t loadedGate[(32 + 1) * 64];
    __shared__ scalar_t loadedValue[(32 + 1) * 64];

    scalar_t acc = 0.0;

    const uint batchSize = gate.size(0);
    const uint innerSize = gate.size(1);

    for (uint innerStart = 0; innerStart < innerSize; innerStart += 32) {
        // Don't overwrite values from last iteration.
        __syncthreads();

        const uint localInnerIndex = innerStart + (threadIdx.x % 32);

        // Load 64 rows of 32 columns.
        for (uint rowIndex = threadIdx.x / 32; rowIndex < 64; rowIndex += 2) {
            const uint outerIndex = rowIndex + blockIdx.x * 64;
            if (outerIndex >= batchSize) {
                break;
            }
            if (localInnerIndex < innerSize) {
                loadedGate[(32 + 1) * rowIndex + threadIdx.x % 32] =
                    gate[outerIndex][localInnerIndex];
                loadedValue[(32 + 1) * rowIndex + threadIdx.x % 32] =
                    value[outerIndex][localInnerIndex];
            }
        }
        __syncthreads();

        // Accumulate 32 values sequentially per thread.
        for (uint i = 0; i < 32; i++) {
            scalar_t gate = loadedGate[(32 + 1) * threadIdx.x + i];
            scalar_t value = loadedValue[(32 + 1) * threadIdx.x + i];
            acc *= gate;
            acc += value;
            // Store the results into loadedGate.
            loadedGate[(32 + 1) * threadIdx.x + i] = acc;
        }
        __syncthreads();

        // Store the values back the same way we loaded them.
        for (uint rowIndex = threadIdx.x / 32; rowIndex < 64; rowIndex += 2) {
            const uint outerIndex = rowIndex + blockIdx.x * 64;
            if (outerIndex >= batchSize) {
                break;
            }
            if (localInnerIndex < innerSize) {
                output[outerIndex][localInnerIndex] =
                    loadedGate[(32 + 1) * rowIndex + threadIdx.x % 32];
            }
        }
    }
}

void coalesced_linear_scan(
    torch::Tensor gate,
    torch::Tensor value,
    torch::Tensor output)
{
    CHECK_INPUT_CUDA(gate);
    CHECK_INPUT_CUDA(value);
    CHECK_INPUT_CUDA(output);
    assert(gate.sizes() == value.sizes());
    assert(gate.sizes() == output.sizes());

    const dim3 threads(64, 1, 1);
    const dim3 blocks(ceil_div(gate.size(0), 64), 1, 1);

    AT_DISPATCH_FLOATING_TYPES(
        gate.scalar_type(),
        "coalesced_linear_scan",
        ([&] { coalesced_linear_scan_kernel<scalar_t><<<blocks, threads>>>(
                   gate.packed_accessor32<scalar_t, 2>(),
                   value.packed_accessor32<scalar_t, 2>(),
                   output.packed_accessor32<scalar_t, 2>()); }));
}

template <typename scalar_t, uint blockSize>
__global__ void __launch_bounds__(blockSize, 1) blocked_linear_scan_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2> gate,
    const torch::PackedTensorAccessor32<scalar_t, 2> value,
    torch::PackedTensorAccessor32<scalar_t, 2> output)
{
    __shared__ scalar_t sharedValues[blockSize];
    __shared__ scalar_t sharedAccs[blockSize];

    const uint innerSize = gate.size(1);
    for (uint i = 0; i < innerSize; i += blockSize) {
        const uint innerIndex = i + threadIdx.x;
        scalar_t loadedGate, loadedValue;
        if (innerIndex < innerSize) {
            loadedGate = gate[blockIdx.x][innerIndex];
            loadedValue = value[blockIdx.x][innerIndex];
        }
        if (i > 0) {
            __syncthreads(); // Finish writing from last loop iteration
            if (threadIdx.x == 0) {
                loadedValue += sharedValues[blockSize - 1] * loadedGate;
            }
            __syncthreads(); // Prevent writing until value is read
        }

        sharedValues[threadIdx.x] = loadedValue;
        sharedAccs[threadIdx.x] = loadedGate;

        scalar_t totalValue = loadedValue;
        scalar_t totalAcc = loadedGate;
        for (uint offset = 1; offset < blockSize; offset *= 2) {
            __syncthreads();
            scalar_t prevValue = 0;
            scalar_t prevAcc = 1;
            if (offset <= threadIdx.x) {
                prevValue = sharedValues[threadIdx.x - offset];
                prevAcc = sharedAccs[threadIdx.x - offset];
            }
            __syncthreads();
            totalValue += totalAcc * prevValue;
            totalAcc *= prevAcc;
            sharedValues[threadIdx.x] = totalValue;
            sharedAccs[threadIdx.x] = totalAcc;
        }

        if (innerIndex < innerSize) {
            output[blockIdx.x][innerIndex] = totalValue;
        }
    }
}

template <typename scalar_t, typename vec_t, uint blockSize>
__global__ void __launch_bounds__(blockSize, 1) vectorized_blocked_linear_scan_kernel(
    const vec_t *gate,
    const vec_t *value,
    vec_t *output,
    uint innerSize)
{
    __shared__ scalar_t sharedValues[blockSize];
    __shared__ scalar_t sharedAccs[blockSize];

    for (uint i = 0; i < innerSize; i += blockSize) {
        const uint innerIndex = i + threadIdx.x;
        vec_t loadedGate, loadedValue;
        if (innerIndex < innerSize) {
            loadedGate = gate[blockIdx.x * innerSize + innerIndex];
            loadedValue = value[blockIdx.x * innerSize + innerIndex];
        }
        if (i > 0) {
            __syncthreads(); // Finish writing from last loop iteration
            if (threadIdx.x == 0) {
                loadedValue.x += sharedValues[blockSize - 1] * loadedGate.x;
            }
            __syncthreads(); // Prevent writing until value is read
        }

        loadedValue.y += loadedValue.x * loadedGate.y;
        loadedValue.z += loadedValue.y * loadedGate.z;
        loadedValue.w += loadedValue.z * loadedGate.w;

        vec_t totalValue = loadedValue;
        vec_t totalAcc = loadedGate;
        totalAcc.y *= totalAcc.x;
        totalAcc.z *= totalAcc.y;
        totalAcc.w *= totalAcc.z;

        sharedValues[threadIdx.x] = totalValue.w;
        sharedAccs[threadIdx.x] = totalAcc.w;

        for (uint offset = 1; offset < blockSize; offset *= 2) {
            __syncthreads();
            scalar_t prevValue = 0;
            scalar_t prevAcc = 1;
            if (offset <= threadIdx.x) {
                prevValue = sharedValues[threadIdx.x - offset];
                prevAcc = sharedAccs[threadIdx.x - offset];
            }
            __syncthreads();
            totalValue.x += totalAcc.x * prevValue;
            totalValue.y += totalAcc.y * prevValue;
            totalValue.z += totalAcc.z * prevValue;
            totalValue.w += totalAcc.w * prevValue;
            totalAcc.x *= prevAcc;
            totalAcc.y *= prevAcc;
            totalAcc.z *= prevAcc;
            totalAcc.w *= prevAcc;
            sharedValues[threadIdx.x] = totalValue.w;
            sharedAccs[threadIdx.x] = totalAcc.w;
        }

        if (innerIndex < innerSize) {
            output[blockIdx.x * innerSize + innerIndex] = totalValue;
        }
    }
}

void blocked_linear_scan(
    torch::Tensor gate,
    torch::Tensor value,
    torch::Tensor output)
{
    CHECK_INPUT_CUDA(gate);
    CHECK_INPUT_CUDA(value);
    CHECK_INPUT_CUDA(output);
    assert(gate.sizes() == value.sizes());
    assert(gate.sizes() == output.sizes());

    if (gate.scalar_type() == torch::ScalarType::Float &&
        gate.size(1) >= 4096 &&
        gate.size(1) % 4 == 0 &&
        ((long)gate.data_ptr()) % 16 == 0) {
        const dim3 threads(1024, 1, 1);
        const dim3 blocks(gate.size(0), 1, 1);
        vectorized_blocked_linear_scan_kernel<float, float4, 1024><<<blocks, threads>>>(
            (const float4 *)gate.data_ptr(),
            (const float4 *)value.data_ptr(),
            (float4 *)output.data_ptr(),
            gate.size(1) / 4);
        return;
    }

#define BLOCKED_LINEAR_SCAN_WITH_SIZE(blockSize)                                     \
    const dim3 threads(blockSize, 1, 1);                                             \
    const dim3 blocks(gate.size(0), 1, 1);                                           \
    AT_DISPATCH_FLOATING_TYPES(                                                      \
        gate.scalar_type(),                                                          \
        "blocked_linear_scan",                                                       \
        ([&] { blocked_linear_scan_kernel<scalar_t, blockSize><<<blocks, threads>>>( \
                   gate.packed_accessor32<scalar_t, 2>(),                            \
                   value.packed_accessor32<scalar_t, 2>(),                           \
                   output.packed_accessor32<scalar_t, 2>()); }));

    if (gate.size(1) <= 128) {
        BLOCKED_LINEAR_SCAN_WITH_SIZE(64);
    } else if (gate.size(1) <= 256) {
        BLOCKED_LINEAR_SCAN_WITH_SIZE(128);
    } else if (gate.size(1) <= 512) {
        BLOCKED_LINEAR_SCAN_WITH_SIZE(256);
    } else if (gate.size(1) <= 1024) {
        BLOCKED_LINEAR_SCAN_WITH_SIZE(512);
    } else {
        BLOCKED_LINEAR_SCAN_WITH_SIZE(1024);
    }
}

template <typename scalar_t>
__global__ void __launch_bounds__(64, 1) simple_linear_scan_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2> gate,
    const torch::PackedTensorAccessor32<scalar_t, 2> output,
    const torch::PackedTensorAccessor32<scalar_t, 2> outGrad,
    torch::PackedTensorAccessor32<scalar_t, 2> gateGradOut,
    torch::PackedTensorAccessor32<scalar_t, 2> valueGradOut)
{
    const uint outerIndex = blockIdx.x * 64 + threadIdx.x;
    if (outerIndex < gate.size(0)) {
        scalar_t doutput = 0.0;
        for (uint i = gate.size(1); i > 0; i--) {
            uint j = i - 1;
            scalar_t prevOutput = 0.0;
            if (j > 0) {
                prevOutput = output[outerIndex][j - 1];
            }
            doutput += outGrad[outerIndex][j];
            valueGradOut[outerIndex][j] = doutput;
            gateGradOut[outerIndex][j] = prevOutput * doutput;
            doutput *= gate[outerIndex][j];
        }
    }
}

void simple_linear_scan_backward(
    torch::Tensor gate,
    torch::Tensor output,
    torch::Tensor outGrad,
    // Writable output tensors:
    torch::Tensor gateGradOut,
    torch::Tensor valueGradOut)
{
    CHECK_INPUT_CUDA(gate);
    CHECK_INPUT_CUDA(output);
    CHECK_INPUT_CUDA(outGrad);
    CHECK_INPUT_CUDA(gateGradOut);
    CHECK_INPUT_CUDA(valueGradOut);
    assert(gate.sizes() == output.sizes());
    assert(gate.sizes() == outGrad.sizes());
    assert(gate.sizes() == gateGradOut.sizes());
    assert(gate.sizes() == valueGradOut.sizes());

    const dim3 threads(64, 1, 1);
    const dim3 blocks(ceil_div(gate.size(0), 64), 1, 1);

    AT_DISPATCH_FLOATING_TYPES(
        gate.scalar_type(),
        "simple_linear_scan_backward",
        ([&] { simple_linear_scan_backward_kernel<scalar_t><<<blocks, threads>>>(
                   gate.packed_accessor32<scalar_t, 2>(),
                   output.packed_accessor32<scalar_t, 2>(),
                   outGrad.packed_accessor32<scalar_t, 2>(),
                   gateGradOut.packed_accessor32<scalar_t, 2>(),
                   valueGradOut.packed_accessor32<scalar_t, 2>()); }));
}

template <typename scalar_t, int blockSize>
__global__ void __launch_bounds__(blockSize, 1) blocked_linear_scan_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2> gate,
    const torch::PackedTensorAccessor32<scalar_t, 2> output,
    const torch::PackedTensorAccessor32<scalar_t, 2> outGrad,
    torch::PackedTensorAccessor32<scalar_t, 2> gateGradOut,
    torch::PackedTensorAccessor32<scalar_t, 2> valueGradOut)
{
    __shared__ scalar_t sharedValues[blockSize];
    __shared__ scalar_t sharedAccs[blockSize];

    const uint innerSize = gate.size(1);
    for (int i = innerSize; i >= 0; i -= blockSize) {
        const int innerIndex = i - blockSize + (int)threadIdx.x;
        scalar_t loadedOutGrad = 0.0;
        if (innerIndex >= 0) {
            loadedOutGrad = outGrad[blockIdx.x][innerIndex];
        }
        scalar_t loadedGate = 0.0;
        if (innerIndex + 1 < innerSize) {
            loadedGate = gate[blockIdx.x][innerIndex + 1];
        }
        scalar_t prevOutput = 0.0;
        if (innerIndex > 0) {
            prevOutput = output[blockIdx.x][innerIndex - 1];
        }
        if (i < innerSize) {
            __syncthreads(); // Finish writing from last loop iteration
            if (threadIdx.x == blockSize - 1) {
                loadedOutGrad += sharedValues[0] * loadedGate;
            }
            __syncthreads(); // Prevent writing until value is read
        }

        sharedValues[threadIdx.x] = loadedOutGrad;
        sharedAccs[threadIdx.x] = loadedGate;

        scalar_t totalValue = loadedOutGrad;
        scalar_t totalAcc = loadedGate;
        for (uint offset = 1; offset < blockSize; offset *= 2) {
            __syncthreads();
            scalar_t prevValue = 0;
            scalar_t prevAcc = 1;
            if (threadIdx.x + offset < blockSize) {
                prevValue = sharedValues[threadIdx.x + offset];
                prevAcc = sharedAccs[threadIdx.x + offset];
            }
            __syncthreads();
            totalValue += totalAcc * prevValue;
            totalAcc *= prevAcc;
            sharedValues[threadIdx.x] = totalValue;
            sharedAccs[threadIdx.x] = totalAcc;
        }

        if (innerIndex < innerSize) {
            valueGradOut[blockIdx.x][innerIndex] = totalValue;
            gateGradOut[blockIdx.x][innerIndex] = totalValue * prevOutput;
        }
    }
}

void blocked_linear_scan_backward(
    torch::Tensor gate,
    torch::Tensor output,
    torch::Tensor outGrad,
    // Writable output tensors:
    torch::Tensor gateGradOut,
    torch::Tensor valueGradOut)
{
    CHECK_INPUT_CUDA(gate);
    CHECK_INPUT_CUDA(output);
    CHECK_INPUT_CUDA(outGrad);
    CHECK_INPUT_CUDA(gateGradOut);
    CHECK_INPUT_CUDA(valueGradOut);
    assert(gate.sizes() == output.sizes());
    assert(gate.sizes() == outGrad.sizes());
    assert(gate.sizes() == gateGradOut.sizes());
    assert(gate.sizes() == valueGradOut.sizes());

#define BLOCKED_LINEAR_SCAN_BACKWARD_WITH_SIZE(blockSize)                                     \
    const dim3 threads(blockSize, 1, 1);                                                      \
    const dim3 blocks(gate.size(0), 1, 1);                                                    \
    AT_DISPATCH_FLOATING_TYPES(                                                               \
        gate.scalar_type(),                                                                   \
        "blocked_linear_scan_backward",                                                       \
        ([&] { blocked_linear_scan_backward_kernel<scalar_t, blockSize><<<blocks, threads>>>( \
                   gate.packed_accessor32<scalar_t, 2>(),                                     \
                   output.packed_accessor32<scalar_t, 2>(),                                   \
                   outGrad.packed_accessor32<scalar_t, 2>(),                                  \
                   gateGradOut.packed_accessor32<scalar_t, 2>(),                              \
                   valueGradOut.packed_accessor32<scalar_t, 2>()); }));

    if (gate.size(1) <= 128) {
        BLOCKED_LINEAR_SCAN_BACKWARD_WITH_SIZE(64);
    } else if (gate.size(1) <= 256) {
        BLOCKED_LINEAR_SCAN_BACKWARD_WITH_SIZE(128);
    } else if (gate.size(1) <= 512) {
        BLOCKED_LINEAR_SCAN_BACKWARD_WITH_SIZE(256);
    } else if (gate.size(1) <= 1024) {
        BLOCKED_LINEAR_SCAN_BACKWARD_WITH_SIZE(512);
    } else {
        BLOCKED_LINEAR_SCAN_BACKWARD_WITH_SIZE(1024);
    }
}

template <typename scalar_t>
__global__ void __launch_bounds__(1024, 1) transposed_linear_scan_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 3> gate,
    const torch::PackedTensorAccessor32<scalar_t, 3> value,
    torch::PackedTensorAccessor32<scalar_t, 3> output,
    uint channelsPerBlock)
{
    __shared__ scalar_t sharedAcc[(32 + 1) * 32];
    __shared__ scalar_t sharedValue[(32 + 1) * 32];
    __shared__ scalar_t prevValue[32];

    const uint batchSize = gate.size(0);
    const uint seqLen = gate.size(1);
    const uint numChannels = gate.size(2);

    // Sizes dependent on how we will coalesce loads
    const uint chunkSize = 1024 / channelsPerBlock;
    const uint blocksPerBatchElem = ceil_div(numChannels, channelsPerBlock);
    const uint batchIdx = blockIdx.x / blocksPerBatchElem;
    const uint startChannel = (blockIdx.x % blocksPerBatchElem) * channelsPerBlock;

    // Indices for loading into shared memory.
    const uint loadChannel = startChannel + (threadIdx.x % channelsPerBlock);
    const uint loadSeqIdx = threadIdx.x / channelsPerBlock;
    const uint storeOffset = shifted_shared_index(threadIdx.x);

    // Indices for gathering our local chunk.
    const uint chunkIndex = threadIdx.x / chunkSize;
    const uint indexInChunk = threadIdx.x % chunkSize;
    const uint loadIndex = shifted_shared_index(chunkIndex + indexInChunk * channelsPerBlock);

    for (uint seqStart = 0; seqStart < seqLen; seqStart += chunkSize) {
        // Don't overwrite values from last iteration.
        __syncthreads();

        // Load global memory into shared memory.
        scalar_t g = 0.0;
        scalar_t v = 0.0;
        if (loadChannel < numChannels && seqStart + loadSeqIdx < seqLen) {
            g = gate[batchIdx][seqStart + loadSeqIdx][loadChannel];
            v = value[batchIdx][seqStart + loadSeqIdx][loadChannel];
        }

        sharedAcc[storeOffset] = g;
        sharedValue[storeOffset] = v;
        __syncthreads();

        // Reduce across each chunk, using a subset of the total
        // shared memory per chunk.
        scalar_t totalValue = sharedValue[loadIndex];
        scalar_t totalAcc = sharedAcc[loadIndex];
        if (seqStart > 0) {
            if (indexInChunk == 0) {
                totalValue += totalAcc * prevValue[chunkIndex];
                sharedValue[loadIndex] = totalValue;
            }
            __syncthreads();
        }
        for (uint offset = 1; offset < chunkSize; offset *= 2) {
            __syncthreads();
            scalar_t prevValue = 0;
            scalar_t prevAcc = 1;
            if (offset <= indexInChunk) {
                const uint prevIndex = shifted_shared_index(chunkIndex + (indexInChunk - offset) * channelsPerBlock);
                prevValue = sharedValue[prevIndex];
                prevAcc = sharedAcc[prevIndex];
            }
            __syncthreads();
            totalValue += totalAcc * prevValue;
            totalAcc *= prevAcc;
            sharedValue[loadIndex] = totalValue;
            sharedAcc[loadIndex] = totalAcc;
        }
        if (indexInChunk == chunkSize - 1) {
            prevValue[chunkIndex] = totalValue;
        }

        // Store the same way we loaded.
        __syncthreads();
        if (loadChannel < numChannels && seqStart + loadSeqIdx < seqLen) {
            output[batchIdx][seqStart + loadSeqIdx][loadChannel] = sharedValue[storeOffset];
        }
    }
}

void transposed_linear_scan(
    torch::Tensor gate,
    torch::Tensor value,
    torch::Tensor output,
    uint channelsPerBlock)
{
    CHECK_INPUT_CUDA(gate);
    CHECK_INPUT_CUDA(value);
    CHECK_INPUT_CUDA(output);
    assert(gate.sizes() == value.sizes());
    assert(gate.sizes() == output.sizes());
    assert(channelsPerBlock == 1 || channelsPerBlock == 2 || channelsPerBlock == 4 || channelsPerBlock == 8 || channelsPerBlock == 16 || channelsPerBlock == 32);

    const uint blocksPerBatchElem = ceil_div(gate.size(2), channelsPerBlock);

    const dim3 threads(1024, 1, 1);
    const dim3 blocks(gate.size(0) * blocksPerBatchElem, 1, 1);

    AT_DISPATCH_FLOATING_TYPES(
        gate.scalar_type(),
        "transposed_linear_scan",
        ([&] { transposed_linear_scan_kernel<scalar_t><<<blocks, threads>>>(
                   gate.packed_accessor32<scalar_t, 3>(),
                   value.packed_accessor32<scalar_t, 3>(),
                   output.packed_accessor32<scalar_t, 3>(),
                   channelsPerBlock); }));
}

template <typename scalar_t>
__global__ void __launch_bounds__(1024, 1) transposed_linear_scan_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 3> gate,
    const torch::PackedTensorAccessor32<scalar_t, 3> output,
    const torch::PackedTensorAccessor32<scalar_t, 3> outGrad,
    torch::PackedTensorAccessor32<scalar_t, 3> gateGradOut,
    torch::PackedTensorAccessor32<scalar_t, 3> valueGradOut,
    uint channelsPerBlock)
{
    __shared__ scalar_t sharedAcc[(32 + 1) * 32];
    __shared__ scalar_t sharedValue[(32 + 1) * 32];
    __shared__ scalar_t prevValue[32];

    const uint batchSize = gate.size(0);
    const uint seqLen = gate.size(1);
    const uint numChannels = gate.size(2);

    // Sizes dependent on how we will coalesce loads
    const uint chunkSize = 1024 / channelsPerBlock;
    const uint blocksPerBatchElem = ceil_div(numChannels, channelsPerBlock);
    const uint batchIdx = blockIdx.x / blocksPerBatchElem;
    const uint startChannel = (blockIdx.x % blocksPerBatchElem) * channelsPerBlock;

    // Indices for loading into shared memory.
    const uint loadChannel = startChannel + (threadIdx.x % channelsPerBlock);
    const uint loadSeqIdx = threadIdx.x / channelsPerBlock;
    const uint storeOffset = shifted_shared_index(threadIdx.x);

    // Indices for gathering our local chunk.
    const uint chunkIndex = threadIdx.x / chunkSize;
    const uint indexInChunk = threadIdx.x % chunkSize;
    const uint loadIndex = shifted_shared_index(chunkIndex + indexInChunk * channelsPerBlock);

    for (uint seqStart = 0; seqStart < seqLen; seqStart += chunkSize) {
        // Don't overwrite values from last iteration.
        __syncthreads();

        // Load global memory into shared memory.
        scalar_t g = 0.0;
        scalar_t v = 0.0;
        if (loadChannel < numChannels) {
            if (seqStart + loadSeqIdx <= seqLen && seqStart + loadSeqIdx > 0) {
                g = gate[batchIdx][seqLen - (seqStart + loadSeqIdx)][loadChannel];
            }
            if (seqStart + loadSeqIdx < seqLen) {
                v = outGrad[batchIdx][seqLen - (seqStart + loadSeqIdx + 1)][loadChannel];
            }
        }

        sharedAcc[storeOffset] = g;
        sharedValue[storeOffset] = v;
        __syncthreads();

        // Reduce across each chunk, using a subset of the total
        // shared memory per chunk.
        scalar_t totalValue = sharedValue[loadIndex];
        scalar_t totalAcc = sharedAcc[loadIndex];
        if (seqStart > 0) {
            if (indexInChunk == 0) {
                totalValue += totalAcc * prevValue[chunkIndex];
                sharedValue[loadIndex] = totalValue;
            }
            __syncthreads();
        }
        for (uint offset = 1; offset < chunkSize; offset *= 2) {
            __syncthreads();
            scalar_t prevValue = 0;
            scalar_t prevAcc = 1;
            if (offset <= indexInChunk) {
                const uint prevIndex = shifted_shared_index(chunkIndex + (indexInChunk - offset) * channelsPerBlock);
                prevValue = sharedValue[prevIndex];
                prevAcc = sharedAcc[prevIndex];
            }
            __syncthreads();
            totalValue += totalAcc * prevValue;
            totalAcc *= prevAcc;
            sharedValue[loadIndex] = totalValue;
            sharedAcc[loadIndex] = totalAcc;
        }
        if (indexInChunk == chunkSize - 1) {
            prevValue[chunkIndex] = totalValue;
        }

        __syncthreads();
        // Use sharedAcc to load the output from the forward pass.
        scalar_t o = 0.0;
        if (loadChannel < numChannels) {
            if (seqStart + loadSeqIdx + 1 < seqLen) {
                o = output[batchIdx][seqLen - (seqStart + loadSeqIdx + 2)][loadChannel];
            }
        }
        sharedAcc[storeOffset] = o;
        __syncthreads();

        if (loadChannel < numChannels && seqStart + loadSeqIdx < seqLen) {
            scalar_t x = sharedValue[storeOffset];
            valueGradOut[batchIdx][seqLen - (seqStart + loadSeqIdx + 1)][loadChannel] = x;
            x *= sharedAcc[storeOffset];
            gateGradOut[batchIdx][seqLen - (seqStart + loadSeqIdx + 1)][loadChannel] = x;
        }
    }
}

void transposed_linear_scan_backward(
    torch::Tensor gate,
    torch::Tensor output,
    torch::Tensor outGrad,
    // Writable output tensors:
    torch::Tensor gateGradOut,
    torch::Tensor valueGradOut,
    // Kernel configuration
    int channelsPerBlock)
{
    CHECK_INPUT_CUDA(gate);
    CHECK_INPUT_CUDA(output);
    CHECK_INPUT_CUDA(outGrad);
    CHECK_INPUT_CUDA(gateGradOut);
    CHECK_INPUT_CUDA(valueGradOut);
    assert(gate.sizes() == output.sizes());
    assert(gate.sizes() == outGrad.sizes());
    assert(gate.sizes() == gateGradOut.sizes());
    assert(gate.sizes() == valueGradOut.sizes());
    assert(channelsPerBlock == 1 || channelsPerBlock == 2 || channelsPerBlock == 4 || channelsPerBlock == 8 || channelsPerBlock == 16 || channelsPerBlock == 32);

    const uint blocksPerBatchElem = ceil_div(gate.size(2), channelsPerBlock);

    const dim3 threads(1024, 1, 1);
    const dim3 blocks(gate.size(0) * blocksPerBatchElem, 1, 1);

    AT_DISPATCH_FLOATING_TYPES(
        gate.scalar_type(),
        "transposed_linear_scan_backward",
        ([&] { transposed_linear_scan_backward_kernel<scalar_t><<<blocks, threads>>>(
                   gate.packed_accessor32<scalar_t, 3>(),
                   output.packed_accessor32<scalar_t, 3>(),
                   outGrad.packed_accessor32<scalar_t, 3>(),
                   gateGradOut.packed_accessor32<scalar_t, 3>(),
                   valueGradOut.packed_accessor32<scalar_t, 3>(),
                   channelsPerBlock); }));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("simple_linear_scan_cpu", &simple_linear_scan_cpu, "Single-threaded CPU linear scan");
    m.def("simple_linear_scan_backward_cpu", &simple_linear_scan_backward_cpu,
          "Single-threaded CPU linear scan backward pass");
    m.def("simple_linear_scan", &simple_linear_scan, "Inefficient linear scan");
    m.def("coalesced_linear_scan", &coalesced_linear_scan, "Linear scan with coalesced loads");
    m.def("blocked_linear_scan", &blocked_linear_scan, "Linear scan with block-level parallelism");
    m.def("simple_linear_scan_backward", &simple_linear_scan_backward, "Inefficient linear scan");
    m.def("blocked_linear_scan_backward", &blocked_linear_scan_backward, "Reverse scan with block-level parallelism");
    m.def("transposed_linear_scan", &transposed_linear_scan, "Scan for NTC instead of NCT");
    m.def("transposed_linear_scan_backward", &transposed_linear_scan_backward, "Scan backward for NTC instead of NCT");
}

#include <torch/extension.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x);

template <typename scalar_t>
__global__ void __launch_bounds__(64, 1) simple_linear_scan_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2> gate,
    const torch::PackedTensorAccessor32<scalar_t, 2> value,
    torch::PackedTensorAccessor32<scalar_t, 2> output)
{
    const uint outerIndex = blockIdx.x * 64 + threadIdx.x;
    if (outerIndex <= gate.size(0)) {
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
    CHECK_INPUT(gate);
    CHECK_INPUT(value);
    CHECK_INPUT(output);
    assert(gate.sizes() == value.sizes());
    assert(gate.sizes() == output.sizes());

    const dim3 threads(64, 1, 1);
    const dim3 blocks(gate.size(0) / 64 + (gate.size(0) % 64 == 0 ? 0 : 1), 1, 1);

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
    CHECK_INPUT(gate);
    CHECK_INPUT(value);
    CHECK_INPUT(output);
    assert(gate.sizes() == value.sizes());
    assert(gate.sizes() == output.sizes());

    const dim3 threads(64, 1, 1);
    const dim3 blocks(gate.size(0) / 64 + (gate.size(0) % 64 == 0 ? 0 : 1), 1, 1);

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

void blocked_linear_scan(
    torch::Tensor gate,
    torch::Tensor value,
    torch::Tensor output)
{
    CHECK_INPUT(gate);
    CHECK_INPUT(value);
    CHECK_INPUT(output);
    assert(gate.sizes() == value.sizes());
    assert(gate.sizes() == output.sizes());

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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("simple_linear_scan", &simple_linear_scan, "Inefficient linear scan");
    m.def("coalesced_linear_scan", &coalesced_linear_scan, "Linear scan with coalesced loads");
    m.def("blocked_linear_scan", &blocked_linear_scan, "Linear scan with block-level parallelism");
}

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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("simple_linear_scan", &simple_linear_scan, "Inefficient linear scan");
}

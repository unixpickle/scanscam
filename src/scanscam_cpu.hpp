#include <torch/extension.h>

#define CHECK_CPU(x) AT_ASSERTM(x.device().is_cpu(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CPU(x);      \
    CHECK_CONTIGUOUS(x);

template <typename scalar_t>
void simple_linear_scan_cpu_kernel(
    const scalar_t *gate,
    const scalar_t *value,
    scalar_t *output,
    int innerSize,
    int outerSize)
{
    for (int i = 0; i < outerSize; i++) {
        const scalar_t *g = &gate[i * innerSize];
        const scalar_t *v = &value[i * innerSize];
        scalar_t *o = &output[i * innerSize];
        scalar_t acc = 0.0;
        for (int j = 0; j < innerSize; j++) {
            acc *= g[j];
            acc += v[j];
            o[j] = acc;
        }
    }
}

void simple_linear_scan_cpu(
    torch::Tensor gate,
    torch::Tensor value,
    torch::Tensor output)
{
    CHECK_INPUT(gate);
    CHECK_INPUT(value);
    CHECK_INPUT(output);
    assert(gate.sizes() == value.sizes());
    assert(gate.sizes() == output.sizes());
    assert(gate.scalar_type() == value.scalar_type());
    assert(gate.scalar_type() == output.scalar_type());

    AT_DISPATCH_FLOATING_TYPES(
        gate.scalar_type(),
        "simple_linear_scan_cpu_kernel",
        ([&] { simple_linear_scan_cpu_kernel<scalar_t>(
                   (const scalar_t *)gate.data_ptr(),
                   (const scalar_t *)value.data_ptr(),
                   (scalar_t *)output.data_ptr(),
                   gate.size(1),
                   gate.size(0)); }));
}

template <typename scalar_t>
void simple_linear_scan_backward_cpu_kernel(
    const scalar_t *gate,
    const scalar_t *output,
    const scalar_t *outGrad,
    scalar_t *gateGradOut,
    scalar_t *valueGradOut,
    int innerSize,
    int outerSize)
{
    for (int i = 0; i < outerSize; i++) {
        scalar_t dOutput = 0.0;
        for (int j = innerSize - 1; j >= 0; j--) {
            scalar_t prevOutput = 0.0;
            if (j > 0) {
                prevOutput = output[i * innerSize + j - 1];
            }
            dOutput += outGrad[i * innerSize + j];
            scalar_t thisGate = gate[i * innerSize + j];
            valueGradOut[i * innerSize + i] = dOutput;
            gateGradOut[i * innerSize + i] = dOutput * prevOutput;
            dOutput *= thisGate;
        }
    }
}

void simple_linear_scan_backward_cpu(
    torch::Tensor gate,
    torch::Tensor output,
    torch::Tensor outGrad,
    // Writable output tensors:
    torch::Tensor gateGradOut,
    torch::Tensor valueGradOut)
{
    CHECK_INPUT(gate);
    CHECK_INPUT(output);
    CHECK_INPUT(outGrad);
    CHECK_INPUT(gateGradOut);
    CHECK_INPUT(valueGradOut);
    assert(gate.sizes() == output.sizes());
    assert(gate.sizes() == outGrad.sizes());
    assert(gate.sizes() == gateGradOut.sizes());
    assert(gate.sizes() == valueGradOut.sizes());
    assert(gate.scalar_type() == output.scalar_type());
    assert(gate.scalar_type() == outGrad.scalar_type());
    assert(gate.scalar_type() == gateGradOut.scalar_type());
    assert(gate.scalar_type() == valueGradOut.scalar_type());

    AT_DISPATCH_FLOATING_TYPES(
        gate.scalar_type(),
        "simple_linear_scan_backward_cpu_kernel",
        ([&] { simple_linear_scan_backward_cpu_kernel<scalar_t>(
                   (const scalar_t *)gate.data_ptr(),
                   (const scalar_t *)output.data_ptr(),
                   (const scalar_t *)outGrad.data_ptr(),
                   (scalar_t *)gateGradOut.data_ptr(),
                   (scalar_t *)valueGradOut.data_ptr(),
                   gate.size(1),
                   gate.size(0)); }));
}

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

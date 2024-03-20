#include "scanscam_cpu.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("simple_linear_scan_cpu", &simple_linear_scan_cpu, "Single-threaded CPU linear scan");
}

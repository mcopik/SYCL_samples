#include <iostream>

#include <SYCL/sycl.hpp>

int main()
{
    std::vector<cl::sycl::device> devices = cl::sycl::device::get_devices( CL_DEVICE_TYPE_ALL /* sycl::info::device_type::all*/);
    std::cout << devices.size() << " devices found by SYCL runtime" << std::endl;
    return 0;
}

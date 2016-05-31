#include <iostream>

#include <SYCL/sycl.hpp>

int main()
{
    // Should be: sycl::info::device_type::all
    std::vector<cl::sycl::device> devices = cl::sycl::device::get_devices( CL_DEVICE_TYPE_ALL );
    std::cout << devices.size() << " devices found by SYCL runtime" << std::endl;
    return 0;
}

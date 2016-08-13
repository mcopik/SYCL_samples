#include <iostream>
#include <algorithm>
#include <SYCL/sycl.hpp>

using namespace cl::sycl;

int main(int argc, char ** argv)
{
    const size_t size = 100;
    int * data = new int[size]();
    
    queue sycl_queue;
    { 
        buffer<int, 1> buf(data, range<1>(size));
        /** Shift by one **/
        auto host_acc_prev = buf.get_access<
                        access::mode::read_write,
                        access::target::host_buffer
                    >();
        sycl_queue.submit([&](handler & cgh) {
            auto acc = buf.get_access<access::mode::write>(cgh);
            cgh.parallel_for<class shift>(range<1>(size),
                [=](id<1> idx) {
                    acc[ idx[0] ] = 1;
                }
            );
        });
        sycl_queue.wait();
        std::cout << "After kernel, before acc creation: " << host_acc_prev[0] << std::endl;
        
        auto host_acc = buf.get_access<
                        access::mode::read,
                        access::target::host_buffer
                    >(); 
        std::cout << "After kernel, after acc creation: " << host_acc_prev[0] << " " << host_acc[0] << std::endl;
        for(int i = 0; i < size; ++i)
            assert(host_acc[i] == 1);
    }
    delete[] data;
    return 0;
}

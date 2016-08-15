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
        sycl_queue.submit([&](handler & cgh) {
            auto acc = buf.get_access<access::mode::write>(cgh);
            cgh.parallel_for<class shift>(range<1>(size),
                [=](id<1> idx) {
                    auto ptr = acc.get_pointer();
                    std::for_each(ptr + idx[0], ptr + idx[0] + 1,
                        [](int & val) { val = 1; });
                }
            );
        });
        
        auto host_acc = buf.get_access<
                        access::mode::read,
                        access::target::host_buffer
                    >(); 
        for(int i = 0; i < size; ++i)
            assert(host_acc[i] == 1);
    }
    delete[] data;
    return 0;
}

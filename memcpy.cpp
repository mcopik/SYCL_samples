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
        sycl_queue.submit([&](handler & cgh) {
            auto acc = buf.get_access<access::mode::write>(cgh);
            cgh.parallel_for<class shift>(range<1>(size),
                [=](id<1> idx) {
                    acc[ idx[0] ] = 1;
                }
            );
        });
        
        {
            // Shift by one on host
            auto host_acc = buf.get_access<
                            access::mode::read_write,
                            access::target::host_buffer
                        >();
            int * input_data = new int[size];
            std::copy(host_acc.get_pointer(),
                        host_acc.get_pointer() + size,
                        input_data); 
            std::for_each(input_data, input_data + size,
                            [](int & val) { ++val; });
            std::copy(input_data, input_data + size,
                host_acc.get_pointer());
            delete[] input_data;
        }
        
        /** Shift by one once more **/
        sycl_queue.submit([&](handler & cgh) {
            auto acc = buf.get_access<access::mode::read_write>(cgh);
            cgh.parallel_for<class shift2>(range<1>(size),
                [=](id<1> idx) {
                    acc[ idx[0] ] += 1;
                }
            );
        });


        auto host_acc = buf.get_access<
                        access::mode::read,
                        access::target::host_buffer
                    >();
        for(int i = 0; i < size; ++i)
            assert(host_acc[i] == 3);
    }
    delete[] data;
    return 0;
}

#include <iostream>
#include <algorithm>
#include <SYCL/sycl.hpp>

using namespace cl::sycl;

class extended_event : public cl::sycl::event
{
    typedef std::function<void(cl_int)> func_t;
    std::vector<func_t> callbacks;
public:

    extended_event(const cl::sycl::event & obj) : cl::sycl::event(obj) {}

    template<typename F>
    void add_callback(F && f)
    {
        callbacks.emplace_back(
            [f](cl_int cmd_exec_status) {
                f(cmd_exec_status);
            }
        );

        clSetEventCallback(
            get(),
            CL_COMPLETE,
            [](cl_event, cl_int exec_status, void* ptr) {
                (*static_cast<func_t*>(ptr))(exec_status);
            },
            &callbacks.back()
        );
    }
};

int main(int argc, char ** argv)
{
    const size_t size = 100;
    int * data = new int[size]();
    
    queue sycl_queue;
    { 
        buffer<int, 1> buf(data, range<1>(size));
        /** Shift by one **/
        auto queue_event = sycl_queue.submit([&](handler & cgh) {
            auto acc = buf.get_access<access::mode::write>(cgh);
            cgh.parallel_for<class shift>(range<1>(size),
                [=](id<1> idx) {
                    acc[ idx[0] ] = 1;
                }
            );
        });
        
        /// Prove we can capture an object succesfully
        /// Lambda convertible to a function ptr requires an empty capture list
        std::string msg("Kernel finished!");
        extended_event queue_event_call(queue_event.get_complete());
        queue_event_call.add_callback(
             [msg](cl_int) { std::cout << msg << std::endl; }
        );

        cl_event marker;
        clEnqueueMarkerWithWaitList(
            sycl_queue.get(),
            0,
            nullptr,
            &marker);
        clSetEventCallback(
            marker,
            CL_COMPLETE,
            [](cl_event, cl_int, void *) { std::cout << "Marker ready!" << std::endl; },
            nullptr
        );
        
        {
            // Shift by one on host
            // Should block and wait for the kernel execution
            auto host_acc = buf.get_access<
                            access::mode::read_write,
                            access::target::host_buffer
                        >();
            std::cout << "Accessor created!" << std::endl;
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

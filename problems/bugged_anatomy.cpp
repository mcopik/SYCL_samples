/// Modified "Anatomy of SYCL application"

#include <iostream>

#include <CL/sycl.hpp>

using namespace cl::sycl;

template<typename Type>
void do_work(Type * data, queue & myQueue)
{
    /// This works
    //buffer<int, 1> resultBuf(data, range<1>(1024));
    /// This doesnt
    /// error: reference to non-static member function must be called on get_access
    buffer<Type, 1> resultBuf(data, range<1>(1024));

    myQueue.submit([&](handler& cgh) {
      auto writeResult = resultBuf.get_access<access::mode::write>(cgh);

      cgh.parallel_for<class DoWork >(range<1>(1024), [=](id<1> idx) {
        writeResult[idx[0]] = static_cast<int>(idx[0]) + 1;
      });
    });
}


int main()
{

    int data[1024];  // initialize data to be worked on

    {
        queue myQueue;
        do_work(data, myQueue);
    }

    for (int i = 0; i < 1024; i++) assert(data[i] == i + 1);

    return 0;
}

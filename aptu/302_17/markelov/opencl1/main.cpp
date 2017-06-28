#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.h>
#include "cl.hpp"

#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <iterator>
#include <string>

int main()
{
   std::vector<cl::Platform> platforms;
   std::vector<cl::Device> devices;
   std::vector<cl::Kernel> kernels;

   try {

      // create platform
      cl::Platform::get(&platforms);
      platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

      // create context
      cl::Context context(devices);

      // create command queue
      cl::CommandQueue queue(context, devices[0]);

      // load input data
      int n;
      int m;
      int N;
      size_t block_size = 16;
      std::vector<float> a;
      std::vector<float> b;
      std::vector<float> c;

      std::cin >> n >> m;
      N = n - n % block_size + block_size;

      a.resize(n * n);
      b.resize(m * m);
      c.resize(n * n, 0);
      for(int i = 0; i < n; ++i) {
          for(int j = 0; j < n; ++j) {
              std::cin >> a[i * n + j];
          }
      }
      for(int i = 0; i < m; ++i) {
          for(int j = 0; j < m; ++j) {
              std::cin >> b[i * m + j];
          }
      }

      // load opencl source
      std::ifstream cl_file("matrix_convolution.cl");
      std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
      cl::Program::Sources source(1, std::make_pair(cl_string.c_str(),
         cl_string.length() + 1));

      // create program
      cl::Program program(context, source);

      // compile opencl source
      std::string arg = "-D BLOCK_SIZE=16";
      program.build(devices, arg.data());

      // allocate device buffer to hold message
      cl::Buffer dev_a(context, CL_MEM_READ_ONLY,  sizeof(float) * n * n);
      cl::Buffer dev_b(context, CL_MEM_READ_ONLY,  sizeof(float) * m * m);
      cl::Buffer dev_c(context, CL_MEM_WRITE_ONLY, sizeof(float) * n * n);

      // copy from cpu to gpu
      queue.enqueueWriteBuffer(dev_a, CL_TRUE, 0, sizeof(float) * n * n, a.data());
      queue.enqueueWriteBuffer(dev_b, CL_TRUE, 0, sizeof(float) * m * m, b.data());

      // load named kernel from opencl source
      cl::Kernel kernel(program, "matrix_conv");
      
      cl::KernelFunctor matrix_conv(kernel, queue, cl::NullRange, cl::NDRange(N, N), cl::NDRange(block_size, block_size));
      matrix_conv(dev_a, dev_b, dev_c, n, m);

      queue.enqueueReadBuffer(dev_c, CL_TRUE, 0, sizeof(float) * n * n, c.data());

      for (size_t i = 0; i < n; ++i)
      {
         for (size_t j = 0; j < n; ++j)
         {
            size_t idx = i * n + j;
            std::cout << c[idx] << " ";
         }
         std::cout << std::endl;
      }
   }
   catch (cl::Error e)
   {
      std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
   }

   return 0;
}

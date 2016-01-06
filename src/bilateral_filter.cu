#include <stdio.h>
#include <stdlib.h>

#include <helper_math.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/opencv.hpp>
#include <vector_types.h>

#define gpuErrchk(ans) { \
  gpuAssert((ans), __FILE__, __LINE__); \
}

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) {
      exit(code);
    }
  }
}

const float SIGMA_COLOR = 30;     //in mm
const float SIGMA_SPACE = 4.5;    // in pixels

__global__ void bilateral_kernel(const cv::gpu::PtrStepSz<float3> src,
                                 cv::gpu::PtrStep<float3> dst,
                                 float sigma_space2_inv_half,
                                 float sigma_color2_inv_half) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  //TODO: bilateral filter

  dst.ptr(y)[x] = src.ptr(y)[x];
}

int divUp(int total, int grain) {
  return (total + grain - 1) / grain;
}

void bilateralFilter(const cv::Mat& src_host, cv::Mat& dst_host) {
  dim3 block(32, 8);
  dim3 grid(divUp(src_host.cols, block.x), divUp(src_host.rows, block.y));

  cudaFuncSetCacheConfig(bilateral_kernel, cudaFuncCachePreferL1);

  cv::gpu::GpuMat src_device(src_host), dst_device(src_host.rows, src_host.cols, src_host.type());

  bilateral_kernel<<<grid, block>>>(
      src_device,
      dst_device,
      0.5f / (SIGMA_SPACE * SIGMA_SPACE),
      0.5f / (SIGMA_COLOR * SIGMA_COLOR)
  );

  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  dst_device.download(dst_host);
}

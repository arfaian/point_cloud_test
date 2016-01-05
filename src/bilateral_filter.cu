#include <stdio.h>
#include <stdlib.h>

#include <helper_math.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <npp.h>
#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

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

__global__ void bilateral_kernel(const cv::gpu::PtrStepSz<ushort> src,
                                 cv::gpu::PtrStep<ushort> dst,
                                 float sigma_space2_inv_half,
                                 float sigma_color2_inv_half) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x >= src.cols || y >= src.rows)
    return;

  const int R = 6;       //static_cast<int>(sigma_space * 1.5);
  const int D = R * 2 + 1;

  int value = src.ptr (y)[x];

  int tx = min (x - D / 2 + D, src.cols - 1);
  int ty = min (y - D / 2 + D, src.rows - 1);

  float sum1 = 0;
  float sum2 = 0;

  for (int cy = max (y - D / 2, 0); cy < ty; ++cy)
  {
    for (int cx = max (x - D / 2, 0); cx < tx; ++cx)
    {
      int tmp = src.ptr (cy)[cx];

      float space2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
      float color2 = (value - tmp) * (value - tmp);

      float weight = __expf (-(space2 * sigma_space2_inv_half + color2 * sigma_color2_inv_half));

      sum1 += tmp * weight;
      sum2 += weight;
    }
  }

  int res = __float2int_rn (sum1 / sum2);
  dst.ptr (y)[x] = max (0, min (res, NPP_MAX_16U));
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

  dst_device.upload(dst_host);
}

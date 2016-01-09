#include <cstdio>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include <helper_cuda.h>
#include <helper_functions.h>
#include <helper_math.h>
#include <helper_string.h>
#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/replace.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <vector_types.h>

#include "node.h"

#define MAX_DEPTH       16
#define INSERTION_SORT  32

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

__device__ void selection_sort(float3* data, int left, int right, int (*compar)(const float3*, const float3*)) {
  for (int i = left ; i <= right ; ++i) {
    float3 min_val = data[i];
    int min_idx = i;

    for (int j = i+1 ; j <= right ; ++j) {
      float3 val_j = data[j];

      if (compar(&val_j, &min_val) == -1) {
        min_idx = j;
        min_val = val_j;
      }
    }

    // Swap the values.
    if (i != min_idx) {
      data[min_idx] = data[i];
      data[i] = min_val;
    }
  }
}

__global__ void cdp_simple_quicksort(float3* data, int left, int right, int (*compar)(const float3*, const float3*), int depth=0) {
  if (depth >= MAX_DEPTH || right-left <= INSERTION_SORT) {
    selection_sort(data, left, right, compar);
    return;
  }

  float3* lptr = data + left;
  float3* rptr = data + right;
  float3  pivot = data[(left + right) / 2];

  while (lptr <= rptr) {
    float3 lval = *lptr;
    float3 rval = *rptr;

    while (compar(&lval, &pivot) == -1) {
      lptr++;
      lval = *lptr;
    }

    while (compar(&lval, &pivot) == 1) {
      rptr--;
      rval = *rptr;
    }

    if (lptr <= rptr) {
      *lptr++ = rval;
      *rptr-- = lval;
    }
  }

  int nright = rptr - data;
  int nleft  = lptr - data;

  if (left < (rptr - data)) {
    cudaStream_t s;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    cdp_simple_quicksort<<<1, 1, 0, s>>>(data, left, nright, compar, depth + 1);
    cudaStreamDestroy(s);
  }

  if ((lptr - data) < right) {
    cudaStream_t s1;
    cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
    cdp_simple_quicksort<<<1, 1, 0, s1>>>(data, nleft, right, compar, depth + 1);
    cudaStreamDestroy(s1);
  }
}

__global__ void bilateral_kernel(const cv::gpu::PtrStepSz<float3> src,
                                 cv::gpu::PtrStep<float3> dst) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  //TODO: bilateral filter

  dst.ptr(y)[x] = src.ptr(y)[x];
}

int divUp(int total, int grain) {
  return (total + grain - 1) / grain;
}

void bilateralFilter(const cv::Mat& src_host, cv::Mat& dst_host, Node* hostKdTree) {
  dim3 block(32, 8);
  dim3 grid(divUp(src_host.cols, block.x), divUp(src_host.rows, block.y));

  cudaFuncSetCacheConfig(bilateral_kernel, cudaFuncCachePreferL1);

  cv::gpu::GpuMat src_device(src_host), dst_device(src_host.rows, src_host.cols, src_host.type());

  //cudaMalloc((void **) deviceKdTree, sizeof(hostKdTree));

  bilateral_kernel<<<grid, block>>>(
      src_device,
      dst_device
  );

  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  dst_device.download(dst_host);
}

int compareX(const float3* a, const float3* b) {
  float arg1 = a->x;
  float arg2 = b->x;
  if (arg1 < arg2) return -1;
  if (arg1 > arg2) return 1;
  return 0;
}

int compareY(const float3* a, const float3* b) {
  float arg1 = a->x;
  float arg2 = b->x;
  if (arg1 < arg2) return -1;
  if (arg1 > arg2) return 1;
  return 0;
}

int compareZ(const float3* a, const float3* b) {
  float arg1 = a->x;
  float arg2 = b->x;
  if (arg1 < arg2) return -1;
  if (arg1 > arg2) return 1;
  return 0;
}

__device__
Node* createKdTree(float3* data, int numPoints, int depth=0) {
  if (numPoints == 0) return NULL;

  int axis = depth % 3;

  int (*compar)(const float3*, const float3*);
  switch (axis) {
    case 0:
      compar = &compareX;
      break;
    case 1:
      compar = &compareY;
      break;
    case 2:
      compar = &compareZ;
      break;
  }

  cdp_simple_quicksort<<<1, 1>>>(data, 0, numPoints - 1, compar);

  int median = numPoints / 2;

  Node* node = new Node;
  node->location = &data[median];
  node->left = createKdTree(data, median, depth + 1);
  node->right = createKdTree(data + (median + 1), numPoints - median - 1, depth + 1);
  return node;
}

__global__
void bilateral_kernel2(float3* pointList, int n) {
  Node* tree = createKdTree(pointList, n);
}

void bilateralFilter2(const float3* pointList, int n) {
  checkCudaErrors(cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, MAX_DEPTH));

  float3* device_pointList;
  cudaMalloc((void **) &device_pointList, n * sizeof(Node));
  cudaMemcpy(device_pointList, pointList, n * sizeof(Node), cudaMemcpyHostToDevice);
  bilateral_kernel2<<<1, 1>>>(device_pointList, n);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
}

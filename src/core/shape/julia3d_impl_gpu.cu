/*

MIT License

Copyright (c) 2019 Zhehang Ding

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#include "julia3d_impl_cpu.h"

#include "core/intersection.h"
#include "core/vector.h"
#include "core/algorithm.h"
#include "core/timer.h"

namespace qjulia {

namespace {
  

struct FractalTestRet {
  bool has_intersection = false;
  Vector3f isect_position;
  Float dist;
};

// TODO: bounding_radius
class Julia3DIntersectKernelGPU {
 public:
  CPU_AND_CUDA Julia3DIntersectKernelGPU(
    Quaternion julia_constant, int max_iterations = 200,
    Float max_magnitude = 10, Float bounding_radius = 3);
  
  CPU_AND_CUDA FractalTestRet SearchIntersection(
    const Vector3f &start, const Vector3f &dir, Float max_dist) const;
    
  CPU_AND_CUDA Vector3f EstimateNormal(const Vector3f &v) const;
  
 private:
  
  CPU_AND_CUDA int Iterate(Quaternion &q, Quaternion &qp) const;
  CPU_AND_CUDA void Iterate(Quaternion &q, int n) const;
  
  Quaternion julia_constant_;
  int max_iterations_ = 200;
  Float max_magnitude_ = 10.0f;
  Float bounding_radius_ = 3.0f;
};

CPU_AND_CUDA Julia3DIntersectKernelGPU::Julia3DIntersectKernelGPU(
    Quaternion julia_constant, int max_iterations,
    Float max_magnitude, Float bounding_radius)
    : julia_constant_(julia_constant), max_iterations_(max_iterations),
      max_magnitude_(max_magnitude), bounding_radius_(bounding_radius) {
}

CPU_AND_CUDA FractalTestRet Julia3DIntersectKernelGPU::SearchIntersection(
    const Vector3f &start, const Vector3f &dir, Float max_dist) const {
  FractalTestRet ret;
  Vector3f p = start;
  while((p - start).Norm() < max_dist) {
    Quaternion q(p[0], p[1], p[2], 0);
    Quaternion qp(1, 0, 0, 0);
    Iterate(q, qp);
    // Esitmate distance.
    Float q_norm = q.Norm(); // TODO: cover 0
    Float qp_norm = qp.Norm();
    Float d = 0.5 * q_norm * std::log(q_norm) / qp_norm;
    if (d < 1e-3) {
      ret.has_intersection = true;
      ret.isect_position = p;
      ret.dist = (p - start).Norm();
      return ret;
    }
    p += dir * d;
  }
  return ret;
}

CPU_AND_CUDA int Julia3DIntersectKernelGPU::Iterate(
    Quaternion &q, Quaternion &qp) const {
  float norm = 0;
  for (int n = 0; n < max_iterations_; ++n) {
    qp *= q;
    qp *= (Float)2;
    q *= q;
    q += julia_constant_;
    norm = q.Norm();
    if (norm > max_magnitude_) {return n;}
  }
  return max_iterations_;
}

CPU_AND_CUDA void Julia3DIntersectKernelGPU::Iterate(Quaternion &q, int n) const {
  for (int i = 0; i < n; ++i) {
    q *= q;
    q += julia_constant_;
  }
}

CPU_AND_CUDA Vector3f Julia3DIntersectKernelGPU::EstimateNormal(
    const Vector3f &v) const {
    
  Float eps = 1e-3;
  Quaternion q(v[0], v[1], v[2], 0);
  Quaternion neighbors[6];
  
  neighbors[0] = q - Quaternion(eps, 0, 0, 0);
  neighbors[1] = q + Quaternion(eps, 0, 0, 0);
  neighbors[2] = q - Quaternion(0, eps, 0, 0);
  neighbors[3] = q + Quaternion(0, eps, 0, 0);
  neighbors[4] = q - Quaternion(0, 0, eps, 0);
  neighbors[5] = q + Quaternion(0, 0, eps, 0);
  
  // NOTE: This implementatiuon is not very stable,
  // with large 'niters' the value goes to inf.
  // May replace this with a more stable method.
  for (int k = 0; k < 6; ++k) {
    int niters = 5;
    Quaternion &nq = neighbors[k];
    Iterate(nq, niters);
    assert(IsFinite(nq));
  }
  
  Float nx = neighbors[1].Norm() - neighbors[0].Norm();
  Float ny = neighbors[3].Norm() - neighbors[2].Norm();
  Float nz = neighbors[5].Norm() - neighbors[4].Norm();
  
  Vector3f normal = Normalize(Vector3f(nx, ny, nz));
  return normal;
}

__device__ __host__ void Func(const Ray &ray, Intersection &isect, Julia3DIntersectKernelGPU &kernel) {
  Float dnear, dfar;
  bool has_roots = IntersectSphere(ray.start, ray.dir, 3, &dnear, &dfar);
  if (has_roots && dfar >= 0) {
    dnear = dnear > 0 ? dnear : 0;
    Vector3f start = ray.start + ray.dir * dnear;
    Vector3f dir = ray.dir;
    Float search_dist = dfar - dnear;
    FractalTestRet ret = kernel.SearchIntersection(start, dir, search_dist);    
    isect.good = ret.has_intersection;
    isect.dist = dnear + ret.dist;
    isect.position = ret.isect_position;
    isect.normal = kernel.EstimateNormal(isect.position);
  } else {
    isect.good = false;
  }
}
  
__global__ void Go(
    const Ray *rays, Intersection *isects,
    int width, int height, Julia3DIntersectKernelGPU kernel) {
  int ir = blockIdx.y * blockDim.y + threadIdx.y;
  int ic = blockIdx.x * blockDim.x + threadIdx.x;
  if (ir >= height || ic >= width) {return;}
  int idx = ir * width + ic;
  const Ray &ray = rays[idx];
  Intersection &isect = isects[idx];
  Func(ray, isect, kernel);
}

}

void Julia3DIntersectGPU(const Array2D<Ray> &rays,
                              Array2D<Intersection> &isects,
                              Quaternion julia_constant, int max_iterations,
                              Float max_magnitude, Float bounding_radius) {
  Timer timer;
  float time_malloc = 0;
  float time_memcpy = 0;
  float time_process = 0;
  
  timer.Start();
  Ray *cu_rays = 0;
  Intersection *cu_isects = 0;
  int total = rays.NumElems();
  cudaMalloc((void**)&cu_rays, sizeof(Ray) * total);
  cudaMalloc((void**)&cu_isects, sizeof(Intersection) * total);
  CHECK_NOTNULL(cu_rays);
  CHECK_NOTNULL(cu_isects);
  time_malloc += timer.End();
  
  timer.Start();
  cudaMemcpy(cu_rays, rays.Data(), sizeof(qjulia::Ray) * total,        
             cudaMemcpyHostToDevice);
  time_memcpy += timer.End();
  
  timer.Start();
  int h = rays.Height();
  int w = rays.Width();
  int bsize = 32;
  int gw = (w + bsize - 1) / bsize;
  int gh = (h + bsize - 1) / bsize;
  dim3 block_size(bsize, bsize);
  dim3 grid_size(gw, gh);
  Julia3DIntersectKernelGPU kernel(
    julia_constant, max_iterations, max_magnitude, bounding_radius);
  Go<<<grid_size, block_size>>>(cu_rays, cu_isects, w, h, kernel);
  cudaDeviceSynchronize();
  time_process += timer.End();
  
  timer.Start();
  cudaMemcpy(isects.Data(), cu_isects,
             sizeof(Intersection) * total, cudaMemcpyDeviceToHost);
  time_memcpy += timer.End();
  
  timer.Start();
  cudaFree(cu_rays);
  cudaFree(cu_isects);
  time_malloc += timer.End();
  
  //printf("malloc:%.2f, time_memcpy:%.2f, time_process:%.2f\n", time_malloc, time_memcpy, time_process);
}

}

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

#include "core/object.h"

#include "core/base.h"
#include "core/shape.h"
#include "core/transform.h"
#include "core/material.h"
#include "core/resource_mgr.h"

namespace qjulia {

Intersection Object::Intersect(const Ray &ray) const {
  Intersection isect;
  CHECK_NOTNULL(shape);
  if (shape == nullptr) {return isect;}
  if (transform != nullptr) {
    Ray ray_local = ray;
    ray_local.start = transform->W2O_Point(ray_local.start);
    ray_local.dir = Normalize(transform->W2O_Vector(ray_local.dir));
    isect = shape->Intersect(ray_local);
    isect.position = transform->O2W_Point(isect.position);
    isect.normal = transform->O2W_Normal(isect.normal);
    isect.dist = transform->O2W_Vector(ray_local.dir * isect.dist).Norm();
  } else {
    isect = shape->Intersect(ray);
  }
  return isect;
}

#define USE_CUDA

#ifdef USE_CUDA

__global__ void IsectCUDAKernel1(Ray *rays, Transform transform, int w, int h) {
  int ir = blockIdx.y * blockDim.y + threadIdx.y;
  int ic = blockIdx.x * blockDim.x + threadIdx.x;
  if (ir >= h || ic >= w) {return;}
  int idx = ir * w + ic;
  Ray &ray = rays[idx];
  ray.start = transform.W2O_Point(ray.start);
  ray.dir = Normalize(transform.W2O_Vector(ray.dir));
}

void IsectCUDA1(const Array2D<Ray> &rays, Array2D<Ray> &rays_out,
                const Transform &transform) {
  const int total = rays.NumElems();
  Ray *cu_rays = 0;
  cudaMalloc((void**)&cu_rays, sizeof(Ray) * total);
  CHECK_NOTNULL(cu_rays);
  cudaMemcpy(cu_rays, rays.Data(), sizeof(Ray) * total,        
             cudaMemcpyHostToDevice);
  int h = rays.Height();
  int w = rays.Width();
  int bsize = 32;
  int gw = (w + bsize) / bsize;
  int gh = (h + bsize) / bsize;
  dim3 block_size(bsize, bsize);
  dim3 grid_size(gw, gh);
  IsectCUDAKernel1<<<grid_size, block_size>>>(cu_rays, transform, w, h);
  cudaDeviceSynchronize();
  cudaMemcpy(rays_out.Data(), cu_rays,
             sizeof(Ray) * total, cudaMemcpyDeviceToHost);
  cudaFree(cu_rays);
}

__global__ void IsectCUDAKernel2(Ray *rays, Intersection *isects, Transform transform, int w, int h) {
  int ir = blockIdx.y * blockDim.y + threadIdx.y;
  int ic = blockIdx.x * blockDim.x + threadIdx.x;
  if (ir >= h || ic >= w) {return;}
  int idx = ir * w + ic;
  const Ray &ray = rays[idx];
  Intersection &isect = isects[idx];
  isect.position = transform.O2W_Point(isect.position);
  isect.normal = transform.O2W_Normal(isect.normal);
  isect.dist = transform.O2W_Vector(ray.dir * isect.dist).Norm();
}

void IsectCUDA2(const Array2D<Ray> &rays, Array2D<Intersection> &isects,
                const Transform &transform) {
  const int total = rays.NumElems();
  Ray *cu_rays = 0;
  Intersection *cu_isects = 0;
  cudaMalloc((void**)&cu_rays, sizeof(Ray) * total);
  cudaMalloc((void**)&cu_isects, sizeof(Intersection) * total);
  CHECK_NOTNULL(cu_rays);
  CHECK_NOTNULL(cu_isects);
  cudaMemcpy(cu_rays, rays.Data(), sizeof(Ray) * total,        
             cudaMemcpyHostToDevice);
  cudaMemcpy(cu_isects, isects.Data(), sizeof(Intersection) * total,        
             cudaMemcpyHostToDevice);
  int h = rays.Height();
  int w = rays.Width();
  int bsize = 32;
  int gw = (w + bsize) / bsize;
  int gh = (h + bsize) / bsize;
  dim3 block_size(bsize, bsize);
  dim3 grid_size(gw, gh);
  IsectCUDAKernel2<<<grid_size, block_size>>>(cu_rays, cu_isects, transform, w, h);
  cudaDeviceSynchronize();
  cudaMemcpy(isects.Data(), cu_isects,
             sizeof(Intersection) * total, cudaMemcpyDeviceToHost);
  cudaFree(cu_rays);
  cudaFree(cu_isects);
}

#endif

void Object::Intersect(const Array2D<Ray> &rays, Array2D<Ray> &rays_cache,
                       Array2D<Intersection> &isects) const {
  CHECK_NOTNULL(shape);
  if (transform) {
#ifdef USE_CUDA
    IsectCUDA1(rays, rays_cache, *transform);
    shape->Intersect(rays_cache, isects);
    IsectCUDA2(rays_cache, isects, *transform);
#else
    for (int i = 0; i < rays.NumElems(); ++i) {
      Ray ray = rays(i);
      ray.start = transform->W2O_Point(ray.start);
      ray.dir = Normalize(transform->W2O_Vector(ray.dir));
      rays_cache(i) = ray;
    }
    shape->Intersect(rays_cache, isects);
    for (int i = 0; i < rays.NumElems(); ++i) {
      auto &isect = isects(i);
      isect.position = transform->O2W_Point(isect.position);
      isect.normal = transform->O2W_Normal(isect.normal);
      isect.dist = transform->O2W_Vector(rays(i).dir * isect.dist).Norm();
    }
#endif
  } else {
    shape->Intersect(rays, isects);
  }
}

bool Object::ParseInstruction(const TokenizedStatement instruction, 
                             const ResourceMgr *resource) {
  // Empty
  if (instruction.size() == 0) {return true;}
  
  // Shape
  if (instruction[0] == GetEntityTypeName(Shape::kType)) {
    return ParseInstruction_Pointer<Shape>(
      instruction, resource, &shape);
  
  // Material
  } else if (instruction[0] == GetEntityTypeName(Material::kType)) {
    return ParseInstruction_Pointer<Material>(
      instruction, resource, &material);
  
  // Transform
  } else if (instruction[0] == GetEntityTypeName(Transform::kType)) {
    return ParseInstruction_Pointer<Transform>(
      instruction, resource, &transform);
  
  } else {
    return UnknownInstructionError(instruction);
  }
}

}

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

namespace qjulia {

namespace {

struct FractalTestRet {
  bool has_intersection = false;
  Vector3f isect_position;
  Float dist;
};
// TODO: bounding_radius
class Julia3DIntersectKernelCPU {
 public:
  Julia3DIntersectKernelCPU(Quaternion julia_constant, int max_iterations,
                            Float max_magnitude, Float bounding_radius);
  
  FractalTestRet SearchIntersection(
    const Vector3f &start, const Vector3f &dir, Float max_dist) const;
    
  Vector3f EstimateNormal(const Vector3f &v) const;
  
 private:
  
  int Iterate(Quaternion &q, Quaternion &qp) const;
  void Iterate(Quaternion &q, int n) const;
  
  Quaternion julia_constant_;
  int max_iterations_ = 200;
  Float max_magnitude_ = 10.0f;
  Float bounding_radius_ = 3.0f;
  
};

Julia3DIntersectKernelCPU::Julia3DIntersectKernelCPU(
    Quaternion julia_constant, int max_iterations,
    Float max_magnitude, Float bounding_radius)
    : julia_constant_(julia_constant), max_iterations_(max_iterations),
      max_magnitude_(max_magnitude), bounding_radius_(bounding_radius) {
}

FractalTestRet Julia3DIntersectKernelCPU::SearchIntersection(
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

int Julia3DIntersectKernelCPU::Iterate(
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

void Julia3DIntersectKernelCPU::Iterate(Quaternion &q, int n) const {
  for (int i = 0; i < n; ++i) {
    q *= q;
    q += julia_constant_;
  }
}

Vector3f Julia3DIntersectKernelCPU::EstimateNormal(
    const Vector3f &v) const {
    
  Float eps = 1e-3;
  Quaternion q(v[0], v[1], v[2], 0);
  std::vector<Quaternion> neighbors(6);
  
  neighbors[0] = q - Quaternion(eps, 0, 0, 0);
  neighbors[1] = q + Quaternion(eps, 0, 0, 0);
  neighbors[2] = q - Quaternion(0, eps, 0, 0);
  neighbors[3] = q + Quaternion(0, eps, 0, 0);
  neighbors[4] = q - Quaternion(0, 0, eps, 0);
  neighbors[5] = q + Quaternion(0, 0, eps, 0);
  
  // NOTE: This implementatiuon is not very stable,
  // with large 'niters' the value goes to inf.
  // May replace this with a more stable method.
  for (int k = 0; k < (int)neighbors.size(); ++k) {
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

void ProcessRay(
    const Ray &ray, Intersection &isect,
    Julia3DIntersectKernelCPU &kernel,
    Float bounding_radius) {
  Float tl, tg;
  Vector3f start = ray.start;
  Vector3f dir = ray.dir;
  if (IntersectSphere(start, dir, bounding_radius, &tl, &tg) && tg >= 0) {
    tl = std::max((Float)0, tl);
    Vector3f bound_start = start + dir * tl;
    FractalTestRet ret = kernel.SearchIntersection(bound_start, dir, tg - tl);
    if (ret.has_intersection) {
      isect.good = true;
      isect.dist = tl + ret.dist;
      isect.position = ret.isect_position;
      isect.normal = kernel.EstimateNormal(isect.position);
    }
  }
}

}

void Julia3DIntersectCPU(
    const Ray &ray, Intersection &isect,
    Quaternion julia_constant, int max_iterations,
    Float max_magnitude, Float bounding_radius) {
  
  Julia3DIntersectKernelCPU kernel(
    julia_constant, max_iterations, max_magnitude, bounding_radius);
  ProcessRay(ray, isect, kernel, bounding_radius);
}

void Julia3DIntersectCPU(
    const Array2D<Ray> &rays, Array2D<Intersection> &isects,
    Quaternion julia_constant, int max_iterations,
    Float max_magnitude, Float bounding_radius) {
  
  Julia3DIntersectKernelCPU kernel(
    julia_constant, max_iterations, max_magnitude, bounding_radius);
  for (int i = 0; i < rays.NumElems(); ++i) {
    Intersection &isect = isects(i);
    const Ray &ray = rays(i);
    ProcessRay(ray, isect, kernel, bounding_radius);
  }
}

}

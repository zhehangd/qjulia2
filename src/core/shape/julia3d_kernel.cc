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

#include "julia3d_kernel.h"

#include "core/intersection.h"
#include "core/vector.h"
#include "core/algorithm.h"

namespace qjulia {
  
#define SHOW_HALF

CPU_AND_CUDA Julia3DIntersectKernel::Julia3DIntersectKernel(
    Quaternion julia_constant) : julia_constant_(julia_constant) {
}

CPU_AND_CUDA FractalTestRet Julia3DIntersectKernel::SearchIntersection(
    const Vector3f &start, const Vector3f &dir, Float max_dist) const {
  FractalTestRet ret;
  Vector3f p = start; // start must be within the bounding sphere
  while((p - start).Norm() < max_dist) {
    if (cross_section_) {
      if (p[2] > 0) {
        if (dir[2] < 0) {
          Float k = - start[2] / dir[2];
          p[0] = start[0] + k * dir[0];
          p[1] = start[1] + k * dir[1];
          p[2] = 0;
        } else {
          break;
        }
      }
    }
    Quaternion q(p[0], p[1], p[2], 0);
    Quaternion qp(1, 0, 0, 0);
    int n = Iterate(q, qp);
    // Esitmate distance.
    Float q_norm = q.Norm(); // TODO: cover 0
    Float qp_norm = qp.Norm();
    Float d = 0.5 * q_norm * std::log(q_norm) / qp_norm;
    if (d < precision_) {
      ret.has_intersection = true;
      ret.isect_position = p;
      ret.dist = (p - start).Norm();
      ret.escape_time = n;
      return ret;
    }
    p += dir * d;
  }
  return ret;
}

CPU_AND_CUDA int Julia3DIntersectKernel::Iterate(
    Quaternion &q, Quaternion &qp) const {
  Float norm = 0;
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

CPU_AND_CUDA void Julia3DIntersectKernel::Iterate(Quaternion &q, int n) const {
  for (int i = 0; i < n; ++i) {
    q *= q;
    q += julia_constant_;
  }
}

CPU_AND_CUDA Vector3f Julia3DIntersectKernel::EstimateNormal(
    const Vector3f &v) const {

  if (cross_section_) {
    if (v[2] >= 0) {return {0, 0, 1};}
  }
    
  Float eps = precision_;
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

CPU_AND_CUDA void IsectJulia3D(
    const Julia3DIntersectKernel &kernel, const Ray &ray, Intersection &isect) {
  Float dnear, dfar;
  bool has_roots = IntersectSphere(ray.start, ray.dir,
                                   kernel.GetBoundingRadius(), &dnear, &dfar);
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
    //Float u = log((Float)ret.escape_time) / log((Float)kernel.GetMaxInterations());
    Float u = (Float)ret.escape_time / (Float)kernel.GetMaxInterations();
    Float u_min = kernel.GetUVBlack();
    Float u_max = kernel.GetUVWhite();
    u = (u - u_min) / (u_max - u_min + 1e-5);
    isect.uv = {u, 0};
  } else {
    isect.good = false;
  }
}

}

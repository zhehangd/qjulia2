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

#ifndef JULIA3D_KERNEL_H_
#define JULIA3D_KERNEL_H_

#include "core/base.h"
#include "core/intersection.h"
#include "core/ray.h"
#include "core/vector.h"

namespace qjulia {

struct FractalTestRet {
  bool has_intersection = false;
  Vector3f isect_position;
  Float dist;
};

class Julia3DIntersectKernel {
 public:
  CPU_AND_CUDA Julia3DIntersectKernel(Quaternion julia_constant);
  
  CPU_AND_CUDA FractalTestRet SearchIntersection(
    const Vector3f &start, const Vector3f &dir, Float max_dist) const;
    
  CPU_AND_CUDA Vector3f EstimateNormal(const Vector3f &v) const;
  
  CPU_AND_CUDA void SetConstant(Quaternion c) {julia_constant_ = c;}
  
  CPU_AND_CUDA Quaternion GetConstant(void) const {return julia_constant_;}
  
  CPU_AND_CUDA void SetPrecision(Float prec) {precision_ = prec;}
  
  CPU_AND_CUDA Float GetPrecision(void) const {return precision_;}
  
  CPU_AND_CUDA void SetMaxInterations(int n) {max_iterations_ = n;}
  
  CPU_AND_CUDA int GetMaxInterations(void) const {return max_iterations_;}
  
  CPU_AND_CUDA void SetEscapeMagnitude(Float r) {max_magnitude_ = r;}
  
  CPU_AND_CUDA Float GetEscapeMagnitude(void) const {return max_magnitude_;}
  
  CPU_AND_CUDA void SetBoundingRadius(Float r) {bounding_radius_ = r;}
  
  CPU_AND_CUDA Float GetBoundingRadius(void) const {return bounding_radius_;}
  
  
 private:
  
  CPU_AND_CUDA int Iterate(Quaternion &q, Quaternion &qp) const;
  
  CPU_AND_CUDA void Iterate(Quaternion &q, int n) const;
  
  Quaternion julia_constant_;
  
  Float precision_ = 1e-3;
  
  int max_iterations_ = 200;
  
  Float max_magnitude_ = 10.0f;
  
  Float bounding_radius_ = 3.0f;
};

CPU_AND_CUDA void IsectJulia3D(
    const Julia3DIntersectKernel &kernel, const Ray &ray, Intersection &isect);

}

#endif

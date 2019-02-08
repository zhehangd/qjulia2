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

#ifndef QJULIA_JULIA3D_H_
#define QJULIA_JULIA3D_H_

#include <vector>
#include <memory>

#include "core/vector.h"
#include "core/shape.h"

namespace qjulia {

class Julia3DShape : public Shape {
 public:
  
  struct FractalTestRet {
    bool has_intersection = false;
    Vector3f isect_position;
    Float dist;
  };
  
  Julia3DShape(Quaternion c) : constant_(c) {}
  Julia3DShape(void) : constant_(0, 0, 0, 0) {}
  
  void SetConstant(Quaternion c) {constant_ = c;}
  
  void UsePreset(int i);
  
  Intersection Intersect(const Ray &ray) const override {
    return Intersect(ray.start, ray.dir);}
  
  Intersection Intersect(const Vector3f &start, const Vector3f &dir) const;
  
  std::string GetImplName(void) const override {return "julia3d";}
  
  SceneEntity* Clone(void) const override {return new Julia3DShape(*this);}
  
  bool ParseInstruction(const TokenizedStatement instruction, 
                        const ResourceMgr *resource) override;
  
 private:
  
  FractalTestRet SearchIntersection(
    const Vector3f &start, const Vector3f &dir, Float max_dist) const;
  
  int GetMaxIterations(void) const {return max_iterations_;}
  // 
  int TestFractal(Quaternion &q) const;
  int TestFractal(Quaternion &q, Quaternion &qp) const;
  
  // Fractal iteration function
  void Iterate(Quaternion &q, Quaternion &qp) const;
  void Iterate(Quaternion &q, Quaternion &qp, int n) const;
  void Iterate(Quaternion &q) const;
  void Iterate(Quaternion &q, int n) const;
  
  Vector3f EstimateNormal(const Vector3f &v) const;
  
  int max_iterations_ = 200;
  Float max_magnitude_ = 10.0f;
  Float bounding_radius_ = 3.0f;
  Quaternion constant_;
};

}

#endif
 

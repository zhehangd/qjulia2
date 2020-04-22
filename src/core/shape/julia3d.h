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
#include "julia3d_kernel.h"

namespace qjulia {

class Julia3DShape : public Shape {
 public:
  
  CPU_AND_CUDA Julia3DShape(Quaternion c) : kernel(c) {}
  CPU_AND_CUDA Julia3DShape(void) : kernel({0, 0, 0, 0}) {}
  
  CPU_AND_CUDA void SetConstant(Quaternion c) {kernel.SetConstant(c);}
  
  CPU_AND_CUDA Quaternion GetConstant(void) const {return kernel.GetConstant();}
  
  CPU_AND_CUDA void SetPrecision(Float prec) {kernel.SetPrecision(prec);}
  
  CPU_AND_CUDA Float GetPrecision(void) const {return kernel.GetPrecision();}
  
  CPU_AND_CUDA void SetMaxInterations(int n) {kernel.SetMaxInterations(n);}
  
  CPU_AND_CUDA int GetMaxInterations(void) const {return kernel.GetMaxInterations();}
  
  CPU_AND_CUDA void SetEscapeMagnitude(Float r) {kernel.SetEscapeMagnitude(r);}
  
  CPU_AND_CUDA Float GetEscapeMagnitude(void) const {return kernel.GetEscapeMagnitude();}
  
  CPU_AND_CUDA void SetBoundingRadius(Float r) {kernel.SetBoundingRadius(r);}
  
  CPU_AND_CUDA Float GetBoundingRadius(void) const {return kernel.GetBoundingRadius();}
  
  CPU_AND_CUDA void SetCrossSectionFlag(bool enable) {kernel.SetCrossSectionFlag(enable);};
  
  CPU_AND_CUDA bool GetCrossSectionFlag(void) const {return kernel.GetCrossSectionFlag();};
  
    CPU_AND_CUDA void SetUVBlack(Float black) {kernel.SetUVBlack(black);}
  
  CPU_AND_CUDA void SetUVWhite(Float white) {kernel.SetUVWhite(white);}
  
  CPU_AND_CUDA Float GetUVBlack(void) const {return kernel.GetUVBlack();}
  
  CPU_AND_CUDA Float GetUVWhite(void) const {return kernel.GetUVWhite();}
  
  CPU_AND_CUDA void UsePreset(int i);
  
  CPU_AND_CUDA Intersection Intersect(const Ray &ray) const override {
    Intersection isect;
    IsectJulia3D(kernel, ray, isect);
    return isect;
  }
  
  void UpdateDevice(Entity *device_ptr) const override;
  
  void Parse(const Args &args, SceneBuilder *build) override;
  
 private:
  
  Julia3DIntersectKernel kernel;
};

}

#endif
 

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

#ifndef QJULIA_LIGHTS_SIMPLE_H_
#define QJULIA_LIGHTS_SIMPLE_H_

#include "core/light.h"

namespace qjulia {

class SunLight : public Light {
 public:
  CPU_AND_CUDA LightRay Li(const Point3f &p) const override;
  
  void UpdateDevice(Entity *device_ptr) const override;
  
  void Parse(const Args &args, SceneBuilder *build) override;
  
  void Save(SceneBuilder*, FnSaveArgs) const override;
  
  Spectrum intensity;
  Vector3f orientation;
};

class PointLight : public Light {
 public:
  CPU_AND_CUDA LightRay Li(const Point3f &p) const override;
  
  void UpdateDevice(Entity *device_ptr) const override;
  
  void Parse(const Args &args, SceneBuilder *build) override;
  
  void Save(SceneBuilder*, FnSaveArgs) const override;
  
  Spectrum intensity;
  Vector3f position;
};

}

#endif

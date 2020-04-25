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

#ifndef QJULIA_MATERIAL_H_
#define QJULIA_MATERIAL_H_

#include "base.h"
#include "entity.h"
#include "spectrum.h"

namespace qjulia {

class Material : public Entity {
 public:
  CPU_AND_CUDA Material(void) {}
  CPU_AND_CUDA Material(Spectrum diffuse) : diffuse(diffuse) {}

#if defined(__CUDACC__)
  CPU_AND_CUDA const Texture* GetTexture(void) const {return texure_device;}
#else
  CPU_AND_CUDA const Texture* GetTexture(void) const {return texure_host;}
#endif  
  void UpdateDevice(Entity *device_ptr) const override;
  
  void Parse(const Args &args, SceneBuilder *build) override;
  
  void Save(SceneBuilder*, FnSaveArgs) const override;
  
  Texture *texure_device = nullptr;
  Texture *texure_host = nullptr;
  
  Spectrum diffuse;
  Float ks = 0.6;
  Float ps = 12;
  Float reflection = 0.2f;
};

}

#endif

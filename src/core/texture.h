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

#ifndef QJULIA_TEXTURE_H_
#define QJULIA_TEXTURE_H_

#include "core/array2d.h"
#include "core/image.h"
#include "core/entity.h"

namespace qjulia {

class Texture : public Entity {
 public:
  CPU_AND_CUDA Texture(void);
  CPU_AND_CUDA ~Texture(void);
  CPU_AND_CUDA void Release(void); // only called in host
  
  void LoadImage(std::string filename);
  void LoadImage(const Image &image);
  
  void UpdateDevice(Entity *device_ptr) const override;
  
  void Parse(const Args &args, SceneBuilder *build) override;
  
  /// @brief Read-only sampling
  CPU_AND_CUDA Vector3f At(Float u, Float v) const;
  CPU_AND_CUDA Vector3f At(Vector2f uv) const {return At(uv[0], uv[1]);}
  
  using HostTextureImg = Array2D<Vector4b>;
  
#if defined(__CUDACC__)
  using DeviceTextureImg = cudaArray;
#else
  using DeviceTextureImg = void;
#endif
  
  HostTextureImg *host_tex_image = nullptr;
  DeviceTextureImg *device_tex_image = nullptr;
  
#if defined(__CUDACC__)
  cudaTextureObject_t tex_object; // cuda
#else
  unsigned long long tex_object;
#endif
};

CPU_AND_CUDA inline Vector3f Texture::At(Float u, Float v) const {
#if defined(__CUDA_ARCH__)
  float4 color4 = tex2D<float4>(tex_object, u, v);
  return {(Float)color4.x, (Float)color4.y, (Float)color4.z};
#else
  int w = host_tex_image->Width();
  int h = host_tex_image->Height();
  int c = static_cast<int>((u * w));
  int r = static_cast<int>((v * h));
  if (c >= w) {c = w - 1;}
  if (c <= 0) {c = 0;}
  if (r >= h) {r = h - 1;}
  if (r <= 0) {r = 0;}
  Vector4b color4 = host_tex_image->At(r, c);
  Vector3f color;
  color[0] = color4[0] / 255.0f;
  color[1] = color4[1] / 255.0f;
  color[2] = color4[2] / 255.0f;
  return color;
#endif
}

}

#endif
 

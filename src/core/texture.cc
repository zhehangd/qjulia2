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

#include "core/texture.h"

#include "core/arg_parse.h"
#include "core/scene_builder.h"

namespace qjulia {


CPU_AND_CUDA Texture::Texture(void) {
}

CPU_AND_CUDA Texture::~Texture(void) {
  Release();
}

void Texture::Release(void) {
  // Only the host copy manages the memory.
  // The device object should do nothing.
#ifndef __CUDA_ARCH__
  if (host_tex_image) {
    delete host_tex_image;
    host_tex_image = nullptr;
  }
  if (device_tex_image) {
    cudaFreeArray(device_tex_image);
    device_tex_image = nullptr;
  }
  if (tex_object) {
    cudaDestroyTextureObject(tex_object);
  }
#endif
}

void Texture::LoadImage(std::string filename) {
  Image image = ReadPNGImage(filename);
  LoadImage(image);
}

void Texture::LoadImage(const Image &image) {
  Release();
  
  auto *tex_image = new HostTextureImg({image.Width(), image.Height()});
  for (int i = 0; i < image.NumElems(); ++i) {
    const auto &src = image.At(i);
    auto &dst = tex_image->At(i);
    dst = {src[0], src[1], src[2], 0};
  }
  host_tex_image = tex_image;

  CHECK(tex_image->Data() != nullptr);
  int w = tex_image->Width();
  int h = tex_image->Height();
  
  // Upload to GPU
  cudaArray *cu_array;
  const cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
  CUDACheckError(__LINE__, cudaMallocArray(&cu_array, &channelDesc, w, h));
  CUDACheckError(__LINE__, cudaMemcpyToArray(
    cu_array, 0, 0, tex_image->Data(), tex_image->NumElems() * 4,
    cudaMemcpyHostToDevice));
  device_tex_image = cu_array;
  
  // Create texture object
  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = cu_array;
  
  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0]   = cudaAddressModeClamp;
  texDesc.addressMode[1]   = cudaAddressModeClamp;
  texDesc.filterMode       = cudaFilterModePoint;
  texDesc.readMode         = cudaReadModeNormalizedFloat;
  texDesc.normalizedCoords = 1;
  
  cudaTextureObject_t texObj = 0;
  CUDACheckError(__LINE__, cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));
  tex_object = texObj;
}

#ifdef WITH_CUDA

KERNEL void UpdateTexture(Entity *dst_b, 
                          Texture::HostTextureImg *host_tex_image,
                          Texture::DeviceTextureImg *device_tex_image,
                          cudaTextureObject_t tex_object) {
  auto *dst = static_cast<Texture*>(dst_b);
  dst->host_tex_image = host_tex_image;
  dst->device_tex_image = device_tex_image;
  dst->tex_object = tex_object; // device code should only use this
}

void Texture::UpdateDevice(Entity *device_ptr) const {
  UpdateTexture<<<1, 1>>>(device_ptr, host_tex_image, device_tex_image, tex_object);
}

#else

void Texture::UpdateDevice(Entity *device_ptr) const {
  (void)device_ptr;
}

#endif

void Texture::Parse(const Args &args, SceneBuilder *build) {
  if (args.size() == 0) {return;}
  if (args[0] == "LoadImage") {
    LoadImage(args[1]);
  } else {
    throw UnknownCommand(args[0]);
  }
}

}

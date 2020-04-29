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

#include "core/developer/default.h"
#include "core/algorithm.h"
#include "core/color.h"

namespace qjulia {

namespace {
#ifdef WITH_CUDA

struct DefaultDevData {
  Size size;
  DefaultDeveloper::CachePixel *cache;
};

KERNEL void RetriveMeta(DefaultDevData *meta, Developer *device_ptr) {
  DefaultDeveloper *dev = static_cast<DefaultDeveloper*>(device_ptr);
  meta->cache = dev->cache_.Data();
  meta->size = dev->cache_.ArraySize();
}

KERNEL void CopyCacheData(DefaultDeveloper::CachePixel *dst, Developer *device_ptr) {
  DefaultDeveloper *dev = static_cast<DefaultDeveloper*>(device_ptr);
  int cache_data_nbtypes = sizeof(DefaultDeveloper::CachePixel) * dev->cache_.ArraySize().Total();
  std::memcpy(dst, dev->cache_.Data(), cache_data_nbtypes);
}
#endif
}

CPU_AND_CUDA void DefaultDeveloper::Develop(const Film &film, float w) {
  for (int i = 0; i < film.NumElems(); ++i) {
    auto &dst = cache_.At(i);
    dst.spectrum += film.At(i).spectrum * w;
    dst.w += w;
  }
}

CPU_AND_CUDA void DefaultDeveloper::Init(Size size) {
  cache_.Resize(size);
  cache_.SetTo({});
}

CPU_AND_CUDA void DefaultDeveloper::Finish(void) {
}

void DefaultDeveloper::RetrieveFromDevice(Developer *device_ptr) {
#ifdef WITH_CUDA
  DefaultDevData *cuda_meta;
  cudaMalloc((void**)&cuda_meta, sizeof(DefaultDevData));
  RetriveMeta<<<1, 1>>>(cuda_meta, device_ptr);
  DefaultDevData meta;
  CUDACheckError(__LINE__,
                 cudaMemcpy(&meta, cuda_meta, sizeof(DefaultDevData),
                            cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize();
  cudaFree(cuda_meta);
  cache_.Resize(meta.size);
  
  int cache_data_nbtypes = sizeof(CachePixel) * meta.size.Total();
  CachePixel *cuda_cache_data;
  cudaMalloc((void**)&cuda_cache_data, cache_data_nbtypes);
  CopyCacheData<<<1, 1>>>(cuda_cache_data, device_ptr);
  
  CUDACheckError(__LINE__,
                 cudaMemcpy(cache_.Data(), cuda_cache_data,
                            cache_data_nbtypes,
                            cudaMemcpyDeviceToHost));
  cudaFree(cuda_cache_data);
#endif
}

void DefaultDeveloper::ProduceImage(RGBImage &image) {
  image.Resize(cache_.ArraySize());
  for (int i = 0; i < image.NumElems(); ++i) {
    auto &src = cache_.At(i);
    image.At(i) = ClipTo8Bit(src.spectrum * (255.0 / src.w));
  }
}

}


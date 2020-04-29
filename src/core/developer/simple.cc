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

#include "core/developer/simple.h"
#include "core/algorithm.h"
#include "core/color.h"

namespace qjulia {

namespace {
#ifdef WITH_CUDA

struct SimpleDevData {
  Size size;
  SimpleDeveloper::CachePixel *cache;
};

KERNEL void RetriveMeta(SimpleDevData *meta, Developer *device_ptr) {
  SimpleDeveloper *dev = static_cast<SimpleDeveloper*>(device_ptr);
  meta->cache = dev->cache_.Data();
  meta->size = dev->cache_.ArraySize();
}
#endif
}


void SimpleDeveloper::Develop(const Film &film, float w) {
  for (int i = 0; i < film.NumElems(); ++i) {
    auto &dst = cache_.At(i);
    dst.spectrum += film.At(i).spectrum * w;
    dst.w += w;
  }
}

void SimpleDeveloper::Init(Size size) {
  cache_.Resize(size);
  cache_.SetTo({});
}
  
void SimpleDeveloper::Finish(void) {
}

void SimpleDeveloper::RetrieveFromDevice(Developer *device_ptr) {
#ifdef WITH_CUDA
  SimpleDevData *cuda_meta;
  cudaMalloc((void**)&cuda_meta, sizeof(SimpleDevData));
  RetriveMeta<<<1, 1>>>(cuda_meta, device_ptr);
  SimpleDevData meta;
  cudaMemcpy(&meta, cuda_meta, sizeof(SimpleDevData), cudaMemcpyDeviceToHost);
  cudaFree(cuda_meta);
  cache_.Resize(meta.size);
  cudaMemcpy(cache_.Data(), meta.cache,
             sizeof(CachePixel) * meta.size.Total(), cudaMemcpyDeviceToHost);
#endif
}

void SimpleDeveloper::ProduceImage(RGBImage &image) {
  image.Resize(cache_.ArraySize());
  for (int i = 0; i < image.NumElems(); ++i) {
    auto &src = cache_.At(i);
    image.At(i) = ClipTo8Bit(src.spectrum * (255.0 / src.w));
  }
}

}


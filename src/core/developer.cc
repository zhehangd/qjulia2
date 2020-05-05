#include "developer.h"

#include "algorithm.h"
#include "array2d.h"

namespace qjulia {

void DeveloperGPU::ProcessSampleFrame(SampleFrame &film, float w) {
  Array2D<Sample> src(film.ArraySize());
  cudaMemcpy(src.Data(), film.Data(),
             sizeof(Sample) * src.NumElems(), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (int i = 0; i < film.NumElems(); ++i) {
    auto &dst = cache_.At(i);
    dst.spectrum += src.At(i).spectrum * w;
    dst.w += w;
  }
}

void DeveloperCPU::ProcessSampleFrame(SampleFrame &film, float w) {
  auto &src = film;
  for (int i = 0; i < film.NumElems(); ++i) {
    auto &dst = cache_.At(i);
    dst.spectrum += src.At(i).spectrum * w;
    dst.w += w;
  }
}

void Developer::Init(Size size) {
  cache_.Resize(size);
  cache_.SetTo({});
}

void Developer::Finish(void) {
  
}

void Developer::ProduceImage(RGBImage &image) {
  image.Resize(cache_.ArraySize());
  for (int i = 0; i < image.NumElems(); ++i) {
    auto &src = cache_.At(i);
    image.At(i) = ClipTo8Bit(src.spectrum * (255.0 / src.w));
  }
}

}

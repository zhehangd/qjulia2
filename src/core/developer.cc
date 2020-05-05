#include "developer.h"

#include "algorithm.h"
#include "array2d.h"
#include "image_io.h"

namespace qjulia {

void DeveloperGPU::ProcessSampleFrame(SampleFrame &cu_film, float w) {
  Array2D<Sample> film(cu_film.ArraySize());
  cudaMemcpy(film.Data(), cu_film.Data(),
             sizeof(Sample) * film.NumElems(), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (int i = 0; i < cu_film.NumElems(); ++i) {
    auto &dst = cache_.At(i);
    auto &src = film.At(i);
    dst.spectrum += src.spectrum * w;
    dst.w += w;
    if (!std::isnan(src.depth)) {
      if (std::isnan(dst.depth)) {
        dst.depth = src.depth * w;
        dst.depth_w = w;
      } else {
        dst.depth += src.depth * w;
        dst.depth_w += w;
      }
    }
  }
}

void DeveloperCPU::ProcessSampleFrame(SampleFrame &film, float w) {
  for (int i = 0; i < film.NumElems(); ++i) {
    auto &dst = cache_.At(i);
    auto &src = film.At(i);
    dst.spectrum += src.spectrum * w;
    dst.w += w;
    if (!std::isnan(src.depth)) {
      if (std::isnan(dst.depth)) {
        dst.depth = src.depth * w;
        dst.depth_w = w;
      } else {
        dst.depth += src.depth * w;
        dst.depth_w += w;
      }
    }
  }
}

void Developer::Init(Size size) {
  cache_.Resize(size);
  cache_.SetTo({});
}

void Developer::Finish(void) {
  for (int i = 0; i < cache_.NumElems(); ++i) {
    auto &pix = cache_.At(i);
    pix.spectrum /= pix.w;
    pix.depth /= pix.depth_w;
  }
  if (options_.flag_save_exposure_map) {
    RGBFloatImage image(cache_.ArraySize());
    for (int i = 0; i < image.NumElems(); ++i) {
      auto &src = cache_.At(i);
      image.At(i) = src.spectrum / src.w;
    }
    Imwrite(options_.exposure_map_filename, image);
  }
  if (options_.flag_save_depth_map) {
    GrayscaleFloatImage image(cache_.ArraySize());
    for (int i = 0; i < image.NumElems(); ++i) {
      auto &src = cache_.At(i);
      image.At(i) = src.depth / src.depth_w;
    }
    Imwrite(options_.depth_map_filename, image);
  }
}

void Developer::ProduceImage(RGBImage &image) {
  image.Resize(cache_.ArraySize());
  for (int i = 0; i < image.NumElems(); ++i) {
    auto &src = cache_.At(i);
    image.At(i) = ClipTo8Bit(src.spectrum * (255.0 / src.w));
  }
}

void Developer::Parse(const Args &args) {
  if (args.size() == 0) {return;}
  if (args[0] == "ExportExposure") {
    CHECK_EQ(args.size(), 2);
    options_.flag_save_exposure_map = true;
    options_.exposure_map_filename = args[1];
  } else if (args[0] == "ExportDepth") {
    options_.flag_save_depth_map = true;
    options_.depth_map_filename = args[1];
  } else {
    LOG(FATAL) << "Unknown command " << args[0];
  }
}

}

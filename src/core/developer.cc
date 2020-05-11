#include "developer.h"

#include "algorithm.h"
#include "arg_parse.h"
#include "array2d.h"
#include "dof_simulator.h"
#include "image_io.h"

namespace qjulia {

void DeveloperGPU::ProcessSampleFrame(SampleFrame &cu_film, float w) {
  // For now GPU impl just copy the data to the host and use the implementation
  // for the host. In the future everything should be implemented in CUDA.
  Array2D<Sample> film(cu_film.ArraySize());
  cudaMemcpy(film.Data(), cu_film.Data(),
             sizeof(Sample) * film.NumElems(), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  ProcessSampleFrameImpl(film, w);
}

void DeveloperCPU::ProcessSampleFrame(SampleFrame &film, float w) {
  ProcessSampleFrameImpl(film, w);
}

void Developer::ProcessSampleFrameImpl(SampleFrame &film, float w) {
  if (options_.flag_use_dof) {
    for (int i = 0; i < film.NumElems(); ++i) {
      auto &src = film.At(i);
      exposure_cache_.At(i) = src.spectrum;
      depth_cache_.At(i) = src.depth;
    }
    dof_.Process(exposure_cache_, depth_cache_, exposure_cache_);
    for (int i = 0; i < film.NumElems(); ++i) {
      auto &dst = cache_.At(i);
      auto &src_exp = exposure_cache_.At(i);
      auto &src_dep = depth_cache_.At(i);
      dst.spectrum += src_exp * w;
      dst.w += w;
      if (!std::isnan(src_dep)) {
        if (std::isnan(dst.depth)) {
          dst.depth = src_dep * w;
          dst.depth_w = w;
        } else {
          dst.depth += src_dep * w;
          dst.depth_w += w;
        }
      }
    }
  } else {
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
}

void Developer::Init(Size size) {
  cache_.Resize(size);
  cache_.SetTo({});
  exposure_cache_.Resize(size);
  depth_cache_.Resize(size);
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

void Developer::SetExposureExport(std::string filename) {
  if (!filename.empty()) {
    options_.flag_save_exposure_map = true;
    options_.exposure_map_filename = filename;
  } else {
    options_.flag_save_exposure_map = false;
    options_.exposure_map_filename = {};
  }
}

void Developer::SetDepthExport(std::string filename) {
  if (!filename.empty()) {
    options_.flag_save_depth_map = true;
    options_.depth_map_filename = filename;
  } else {
    options_.flag_save_depth_map = false;
    options_.depth_map_filename = {};
  }
}

void Developer::EnableDOF(bool enable) {
  options_.flag_use_dof = enable;
}

void Developer::Parse(const Args &args) {
  if (args.size() == 0) {return;}
  if (args[0] == "ExportExposure") {
    CHECK_EQ(args.size(), 2);
    SetExposureExport(args[1]);
  } else if (args[0] == "ExportDepth") {
    SetDepthExport(args[1]);
  } else if (args[0] == "EnableDOF") {
    bool val;
    ParseArg(args[1], val);
    EnableDOF(val);
  } else {
    LOG(FATAL) << "Unknown command " << args[0];
  }
}

}

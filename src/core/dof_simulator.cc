#include "dof_simulator.h"

#include <atomic>
#include <cmath>
#include <thread>

#include "arg_parse.h"
#include "timer.h"

namespace qjulia {

CPU_AND_CUDA int BlurKernel::CalculateHalfKernelSize(float radius) {
  return std::floor(radius);
}

CPU_AND_CUDA int BlurKernel::CalculateKernelSize(float radius) {
  return 2 * CalculateHalfKernelSize(radius) + 1;
}

CPU_AND_CUDA Float BlurKernel::Sample(float x, float y, float radius) {
  Float d2 = x * x + y * y;
  Float r2 = radius * radius;
  return (d2 <= r2) / (r2 * kPi + (Float)0.25);
  // target:
  // 0.5 -> 1
  // 1.5 -> 5
  // 2.5 -> 9
  // 3.5 -> 29
}

void DOFSimulator::Parse(const Args &args) {
  if (args.size() == 0) {return;}
  if (args[0] == "SetDepthPrecision") {
    Float val;
    ParseArg(args[1], val);
    SetDepthPrecision(val);
  } else if (args[0] == "SetBlurStrength") {
    Float val;
    ParseArg(args[1], val);
    SetBlurStrength(val);
  } else if (args[0] == "SetFarDOFLimit") {
    Float val;
    ParseArg(args[1], val);
    SetFarDOFLimit(val);
  } else if (args[0] == "SetFarHyperfocalDepth") {
    Float val;
    ParseArg(args[1], val);
    SetFarHyperfocalDepth(val);
  } else if (args[0] == "SetOcclusionAttenuation") {
    Float val;
    ParseArg(args[1], val);
    SetOcclusionAttenuation(val);
  } else {
    LOG(FATAL) << "Unknown command " << args[0];
  }
}

void DOFSimulatorCPU::MultithreadCall(std::function<void(int,int)> fn) {
  const int width = data_.width;
  const int height = data_.height;
  
  if (data_.max_receptive_field_size * 8 <= data_.height) {
    int num_threads = std::thread::hardware_concurrency();
    std::atomic<int> atomic_row(0);
    std::vector<std::thread> threads;
    
    auto RunThread = [&](void) {
      while(true) {
        int row_start = atomic_row.fetch_add(data_.max_receptive_field_size + 1);
        if (row_start >= height) {break;}
        int row_end = std::min(height, row_start + data_.max_receptive_field_size / 2 + 1);
        for (int r = row_start; r < row_end; ++r) {
          //LOG_IF(INFO, r % 100 == 0) << "ROW #" << r;
          for (int c = 0; c < width; ++c) {
            fn(r, c);
          }
        }
      }
    };
    
    for (int i = 0; i < num_threads; ++i) {threads.emplace_back(RunThread);}
    for (auto &thread : threads) {thread.join();}
    threads.clear();
    
    atomic_row = data_.max_receptive_field_size / 2 + 1;
    for (int i = 0; i < num_threads; ++i) {threads.emplace_back(RunThread);}
    for (auto &thread : threads) {thread.join();}
    threads.clear();
  } else {
    for (int r = 0; r < height; ++r) {
      //LOG_IF(INFO, r % 100 == 0) << "ROW #" << r;
      for (int c = 0; c < width; ++c) {
        fn(r, c);
      }
    }
  }
}

void DOFSimulatorCPU::WindowCall(int sr, int sc, Float blur_radius, std::function<void(int,int)> fn) {
  int kernel_hsize = BlurKernel::CalculateHalfKernelSize(blur_radius);
  int win_r_start = std::max(sr - kernel_hsize, 0);
  int win_c_start = std::max(sc - kernel_hsize, 0);
  int win_r_end = std::min(sr + kernel_hsize + 1, data_.height); // [start, end)
  int win_c_end = std::min(sc + kernel_hsize + 1, data_.width);
  for (int wr = win_r_start; wr < win_r_end; ++wr) {
    for (int wc = win_c_start; wc < win_c_end; ++wc) {
      fn(wr, wc);
    }
  }
}

Float DOFSimulatorCPU::DepthToBlurRadius(Float depth) {
  if (depth <= data_.far_dof_limit) {
    return 0;
  } else if (depth < data_.far_hyperfocal_depth) {
    return data_.max_blur_raidus * (depth - data_.far_dof_limit) / (data_.far_hyperfocal_depth - data_.far_dof_limit);
  } else {
    return data_.max_blur_raidus;
  }
}

Float DOFSimulatorCPU::StrengthToBlurRadius(Float strength) {
  return strength * 0.01 * data_.minwh;
}

void DOFSimulatorCPU::Analyze(RGBFloatImage exposure, GrayscaleFloatImage depth) {
  CHECK_EQ(exposure.ArraySize(), depth.ArraySize());
  data_.width = exposure.Width();
  data_.height = exposure.Height();
  data_.minwh = std::min(data_.width, data_.height);
  data_.cache_meta.Resize(exposure.ArraySize());
  data_.cache_meta.SetTo({});
  data_.max_blur_raidus = StrengthToBlurRadius(data_.blur_strength);
  
  data_.max_receptive_field_size = BlurKernel::CalculateKernelSize(data_.max_blur_raidus);
  
  MultithreadCall([&](int sr, int sc) {
    auto &meta = data_.cache_meta.At(sr, sc);
    meta.exposure = exposure.At(sr, sc);
    meta.depth = depth.At(sr, sc);
    meta.blur_radius = DepthToBlurRadius(meta.depth);
    meta.compensation = 0;
    
    //CHECK(!std::isnan(meta.exposure[0]));
    if (std::isnan(meta.depth)) {return;}
    
    CHECK_GE(meta.blur_radius, 0);
    
    if (meta.blur_radius >= 1) {
      WindowCall(sr, sc, meta.blur_radius, [&](int wr, int wc) {
        CHECK_GE(meta.blur_radius, 0);
        Float ww = BlurKernel::Sample(wr - sr, wc - sc, meta.blur_radius);
        CHECK_GE(ww, 0);
        CHECK_LE(ww, 1);
        
        Float wd = depth.At(wr, wc);
        
        // Count the area where the bokeh generated by the source pixel
        // is occuluded by the foreground. The less the area, the more
        // gain we should compensate the bokeh.
        if (meta.depth <= (wd + data_.depth_precision) || std::isnan(wd)) {
          meta.compensation += ww;
        }
        
        // Calculate the occlusion applied to the source pixel.
        // We make an approximation that only pixel in DOF will
        // make this occlusion. Otherwise we will have O(n^4)
        // complexity.
        if (wd <= data_.far_dof_limit) {
          Float dww = BlurKernel::Sample(sr - wr, sc - wc, DepthToBlurRadius(wd));
          meta.opacity_far += dww;
        }
      });
      meta.compensation = 1.0f / (meta.compensation + data_.occlusion_attenuation);
    } else {
      meta.compensation = 1.0f;
      meta.opacity_far = 1.0;
    }
  });
}

void DOFSimulatorCPU::Compose(RGBFloatImage &dst) {
  dst.Resize(data_.cache_meta.ArraySize());
  dst.SetTo({});
  MultithreadCall([&](int sr, int sc) {
    auto &smeta = data_.cache_meta.At(sr, sc);
    // DEBUG: Visualize opacity
    //dst.At(sr, sc) = {smeta.opacity, smeta.opacity, smeta.opacity};
    //return;
    
    if (std::isnan(smeta.depth)) {return;}
    
    int kernel_hsize = BlurKernel::CalculateHalfKernelSize(smeta.blur_radius);
    if (kernel_hsize == 0) {
      dst.At(sr, sc) += smeta.exposure;
    } else {
      WindowCall(sr, sc, smeta.blur_radius, [&](int wr, int wc) {
        const auto &wmeta = data_.cache_meta.At(wr, wc);
        Float ww = BlurKernel::Sample(wr - sr, wc - sc, smeta.blur_radius);
        auto val = smeta.exposure * ww * smeta.compensation;
        //CHECK(!std::isnan(smeta.compensation));
        //CHECK(!std::isnan(ww));
        //CHECK(!std::isnan(smeta.exposure[0]));
        //CHECK(!std::isnan(val[0]));
        
        // Approximation: only pixel behind DOF will 
        // be occluded by pixels in DOF
        Float opacity = wmeta.opacity_far * (smeta.depth >= data_.far_dof_limit);
        
        // The following code shows how to accurately calculate the occlusion
        // You should not use it because it is super slow.
        // --------------------------------------
        // opacity = 0;
        // WindowCall(wr, wc, data_.max_blur_raidus, [&](int wr2, int wc2) {
        //   auto &wmeta2 = data_.cache_meta.At(wr2, wc2);
        //   if (wmeta2.depth <= (smeta.depth - data_.depth_precision)) {
        //     Float ww2 = BlurKernel::Sample(
        //       wr - wr2, wc - wc2, wmeta2.blur_radius);
        //     opacity += ww2;
        //   }
        // });
        // --------------------------------------
        //CHECK(!std::isnan(opacity));
        dst.At(wr, wc) += val * (1 - std::min((Float)1.0, opacity));
        //CHECK(!std::isnan(dst.At(wr, wc)[0]));
      });
    }
  });
}

void DOFSimulatorCPU::Process(RGBFloatImage exposure, GrayscaleFloatImage depth,
                              RGBFloatImage &dst) {
  Timer  timer;
  timer.Start();
  Analyze(exposure, depth);
  float time_analysis = timer.End();
  
  timer.Start();
  Compose(dst);
  float time_composition = timer.End();
  
  LOG(INFO) << "analysis=" << time_analysis << ", composition=" << time_composition;
}

}

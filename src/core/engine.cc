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

#include "core/engine.h"

#include <atomic>
#include <thread>

#include <glog/logging.h>

#include "core/scene_builder.h"

#include "core/algorithm.h"
#include "core/camera.h"
#include "core/film.h"
#include "core/integrator.h"
#include "core/light.h"
#include "core/object.h"
#include "core/scene.h"
#include "core/timer.h"
#include "core/vector.h"

#include "core/integrator/default.h"
#include "core/integrator/normal.h"
#include "core/developer/default.h"

namespace qjulia {

struct AASample {
  AASample(Float x, Float y, Float w) : offset(x, y), w(w) {}
  Vector2f offset;
  Float w = 1;
};

const AASample static_aa_samples[6] = {
  AASample(-0.52f,  0.38f, 0.128f), AASample( 0.41f,  0.56f, 0.119f),
  AASample( 0.27f,  0.08f, 0.294f), AASample(-0.17f, -0.29f, 0.249f),
  AASample( 0.58f, -0.55f, 0.104f), AASample(-0.31f, -0.71f, 0.106f),
};

#ifdef WITH_CUDA

struct CUDAImpl {
  
  CUDAImpl(void);
  ~CUDAImpl(void) {}
  
  void Render(SceneBuilder &build, const RenderOptions &options, Image &image);
  
  int cu_film_data_size_ = 0;
  
  std::unique_ptr<Sample, void(*)(Sample*)> cu_film_data_ {
    nullptr, [](Sample *p) {cudaFree(p);}};
};

CUDAImpl::CUDAImpl(void) {
}

//CPU_AND_CUDA void MergeSample

KERNEL void GPUKernel(Film film, Scene scene, AASample aa) {
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (!film.IsValidCoords(r, c)) {return;}
  int i = film.GetIndex(r, c);
  DefaultIntegrator integrator;
  
  Float x, y;
  Float fr = r + aa.offset[0];
  Float fc = c + aa.offset[1];
  film.GenerateCameraCoords(fr, fc, &x, &y);
  Ray ray = scene.GetCamera()->CastRay({x, y});
  film(i) = integrator.Li(ray, scene);
}

void CUDAImpl::Render(SceneBuilder &build,
                      const RenderOptions &options, Image &image) {
  int w = image.Width();
  int h = image.Height();
  const int film_data_bytes = w * h * sizeof(Sample);
  if (!cu_film_data_ || (cu_film_data_size_ != w * h)) {
    cu_film_data_.reset();
    Sample *spectrum_ptr;
    CUDACheckError(__LINE__, cudaMalloc((void**)&spectrum_ptr, film_data_bytes));
    CHECK_NOTNULL(spectrum_ptr);
    cu_film_data_.reset(spectrum_ptr);
    cu_film_data_size_ = w * h;
  }
  
  Film film(image.ArraySize());
  Film cu_film(cu_film_data_.get(), w, h);  
  
  BuildSceneParams params;
  params.cuda = true;
  Scene scene = build.BuildScene(params);
  
  int bsize = 16;
  int gw = (w + bsize - 1) / bsize;
  int gh = (h + bsize - 1) / bsize;
  dim3 block_size(bsize, bsize);
  dim3 grid_size(gw, gh);
  
  DefaultDeveloper developer;
  developer.Init(image.ArraySize());
  
  if (options.antialias) {
    for (int i = 0; i < 6; ++i) {
      GPUKernel<<<grid_size, block_size>>>(cu_film, scene, static_aa_samples[i]);
      CUDACheckError(__LINE__, cudaMemcpy(film.Data(), cu_film_data_.get(),
                    film_data_bytes, cudaMemcpyDeviceToHost));
      cudaDeviceSynchronize();
      developer.Develop(film, static_aa_samples[i].w);
    }
  } else {
    GPUKernel<<<grid_size, block_size>>>(cu_film, scene, AASample(0, 0, 1));
    CUDACheckError(__LINE__, cudaMemcpy(film.Data(), cu_film_data_.get(),
                  film_data_bytes, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    developer.Develop(film, 1.0);
  }
  
  
  developer.Finish(image);
}

#endif

struct CPUImpl {
  
  CPUImpl(void) {}
  ~CPUImpl(void) {}
  
  void Render(SceneBuilder &build, const RenderOptions &options, Image &image);
};

void CPUImpl::Render(
    SceneBuilder &build, const RenderOptions &options, Image &image) {
  BuildSceneParams params;
  params.cuda = false;
  Scene scene = build.BuildScene(params);
  
  DefaultIntegrator integrator;
  const Camera *camera = scene.GetCamera();
  Film film(image.ArraySize());
  
  int num_threads = options.num_threads;
  if (num_threads < 0) {num_threads = std::thread::hardware_concurrency();}
  
  std::vector<AASample> aa_filters;
  if (options.antialias) {
    for (int i = 0; i < 6; ++i) {aa_filters.push_back(static_aa_samples[i]);}
  } else {
    aa_filters.push_back(AASample(0, 0, 1));
  }
  
  DefaultDeveloper developer;
  developer.Init(image.ArraySize());
  
  for (const auto &aa : aa_filters) {
  
    std::atomic<int> row_cursor(0);
    auto RunThread = [&](void) {
      while(true) {
        int r = row_cursor++;
        if (r >= film.Height()) {break;}
        auto *row = film.Row(r);
        for (int c = 0; c < film.Width(); ++c) {
          Sample &sample = row[c];
          sample.depth = 0;
          Float x, y, fr, fc;
          fr = r + aa.offset[0];
          fc = c + aa.offset[1];
          film.GenerateCameraCoords(fr, fc, &x, &y);
          Vector2f p = Vector2f(x, y);
          Ray ray = camera->CastRay(p);
          sample = integrator.Li(ray, scene);
        }
      }
    };
    
    if (num_threads == 0) {
      RunThread();
    } else {
      std::vector<std::thread> threads;
      for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(RunThread);
      }
      for (auto &thread : threads) {thread.join();}
    }
    developer.Develop(film, aa.w);
  }
  developer.Finish(image);
}

class RTEngine::Impl {
 public:
  Impl(void);
  
  void Render(SceneBuilder &build, const RenderOptions &options, Image &image);
#ifdef WITH_CUDA  
  CUDAImpl cuda_impl;
#endif
  CPUImpl cpu_impl;
};


RTEngine::Impl::Impl(void) {
}

void RTEngine::Impl::Render(
    SceneBuilder &build,
    const RenderOptions &options, Image &image) {
  if (options.cuda) {
#ifdef WITH_CUDA
    cuda_impl.Render(build, options, image);
#else
    LOG(FATAL) << "qjulia2 is not compiled with CUDA support.";
#endif
  } else {
    cpu_impl.Render(build, options, image);
  }
}

RTEngine::RTEngine(void) : impl_(new Impl()) {
  
}

RTEngine::~RTEngine(void) {
}

void RTEngine::Render(SceneBuilder &build,
                      const RenderOptions &options, Image &image) {
  Timer timer;
  timer.Start();
  impl_->Render(build, options, image);
  last_render_time_ = timer.End();
}

}

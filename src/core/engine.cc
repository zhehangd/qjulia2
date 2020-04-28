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
#include "core/developer/default.h"

namespace qjulia {

struct AAFilter {
  AAFilter(Float x, Float y, Float w) : offset(x, y), w(w) {}
  Vector2f offset;
  Float w = 1;
};

std::vector<AAFilter> GenerateSSAAFilters(AAOption opt) {
  std::vector<AAFilter> filters;
  if (opt == AAOption::kOff) {
    filters.emplace_back(0, 0, 1);
  } else if (opt == AAOption::kSSAA6x) {
    filters.emplace_back(-0.52f,  0.38f, 0.128f);
    filters.emplace_back( 0.41f,  0.56f, 0.119f);
    filters.emplace_back( 0.27f,  0.08f, 0.294f);
    filters.emplace_back(-0.17f, -0.29f, 0.249f);
    filters.emplace_back( 0.58f, -0.55f, 0.104f);
    filters.emplace_back(-0.31f, -0.71f, 0.106f);
  } else if (opt == AAOption::kSSAA64x) {
    for (int r = 0; r < 8; ++r) {
      for (int c = 0; c < 8; ++c) {
        float fr = r / 8.0f;
        float fc = c / 8.0f;
        filters.emplace_back(fr, fc, 1.0 / 64.0);
      }
    }
  } else if (opt == AAOption::kSSAA256x) {
    for (int r = 0; r < 16; ++r) {
      for (int c = 0; c < 16; ++c) {
        float fr = r / 16.0f;
        float fc = c / 16.0f;
        filters.emplace_back(fr, fc, 1.0 / 256.0);
      }
    }
  }
  return filters;
}



const AAFilter static_aa_samples[6] = {
  AAFilter(-0.52f,  0.38f, 0.128f), AAFilter( 0.41f,  0.56f, 0.119f),
  AAFilter( 0.27f,  0.08f, 0.294f), AAFilter(-0.17f, -0.29f, 0.249f),
  AAFilter( 0.58f, -0.55f, 0.104f), AAFilter(-0.31f, -0.71f, 0.106f),
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

KERNEL void GPUKernel(Film film, Scene scene, AAFilter aa) {
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
    Sample *p = nullptr;
    CUDACheckError(__LINE__, cudaMalloc((void**)&p, film_data_bytes));
    CHECK_NOTNULL(p);
    cu_film_data_.reset(p);
    cu_film_data_size_ = w * h;
  }
  
  // Right now film data must be allocated every time.
  // I expect in the future film development can be entirely done
  // in GPU, which means we don't need a host film anymore.
  Film film(image.ArraySize());
  Film cu_film(cu_film_data_.get(), w, h);  
  CHECK(!cu_film.HasOwnership());
  
  // Scene is only a simple structure that contains
  // the pointers to the world and the camera entities.
  // So we can pass the object to CUDA kernel as a parameter,
  // as long as the entities pointers are for the device.
  BuildSceneParams params;
  params.cuda = true;
  Scene scene = build.BuildScene(params);
  
  // I have little knowledge of setting the optimal block size.
  // 16 works for my PC.
  int bsize = 16;
  int gw = (w + bsize - 1) / bsize;
  int gh = (h + bsize - 1) / bsize;
  dim3 block_size(bsize, bsize);
  dim3 grid_size(gw, gh);
  
  Developer &developer = *CHECK_NOTNULL(options.developer);
  developer.Init(image.ArraySize());
  
  std::vector<AAFilter> aa_filters = GenerateSSAAFilters(options.aa);
  for (int i = 0; i < aa_filters.size(); ++i) {
    CHECK_NOTNULL(film.Data());
    CHECK_NOTNULL(cu_film_data_.get());
    GPUKernel<<<grid_size, block_size>>>(cu_film, scene, aa_filters[i]);
    CHECK(film.Data() && cu_film_data_.get());
    CUDACheckError(__LINE__, cudaMemcpy(film.Data(), cu_film_data_.get(),
                  film_data_bytes, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    developer.Develop(film, aa_filters[i].w);
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
  
  std::vector<AAFilter> aa_filters = GenerateSSAAFilters(options.aa);
  Developer &developer = *(options.developer);
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

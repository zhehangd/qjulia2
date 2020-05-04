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
#include "core/world.h"
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
  
  Developer* Render(SceneBuilder &build, const RenderOptions &options);
  
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
  
  DefaultIntegrator default_integrator;
  Integrator *integrator = scene.GetIntegrator();
  if (integrator == nullptr) {integrator = &default_integrator;}
  
  Float x, y;
  Float fr = r + aa.offset[0];
  Float fc = c + aa.offset[1];
  film.GenerateCameraCoords(fr, fc, &x, &y);
  Ray ray = scene.GetCamera()->CastRay({x, y});
  film(i) = integrator->Li(ray, scene);
}

KERNEL void InitDeveloper(Developer *cu_developer, Size size) {
  cu_developer->Init(size);
}

KERNEL void Develop(Developer *cu_developer, Film cu_film, Float w) {
  cu_developer->Develop(cu_film, w);
}

KERNEL void FinishDeveloper(Developer *cu_developer) {
  cu_developer->Finish();
}

Developer* CUDAImpl::Render(SceneBuilder &build, const RenderOptions &options) {
  Size size = options.size;
  const int film_data_bytes = size.Total() * sizeof(Sample);
  if (!cu_film_data_ || (cu_film_data_size_ != size.Total())) {
    cu_film_data_.reset();
    Sample *p = nullptr;
    CUDACheckError(__LINE__, cudaMalloc((void**)&p, film_data_bytes));
    CHECK_NOTNULL(p);
    cu_film_data_.reset(p);
    cu_film_data_size_ = size.Total();
  }
  
  // Right now film data must be allocated every time.
  // I expect in the future film development can be entirely done
  // in GPU, which means we don't need a host film anymore.
  Film film(size);
  Film cu_film(cu_film_data_.get(), size.width, size.height);  
  CHECK(!cu_film.HasOwnership());
  
  // Scene is only a simple structure that contains
  // the pointers to the world and the camera entities.
  // So we can pass the object to CUDA kernel as a parameter,
  // as long as the entities pointers are for the device.
  BuildSceneParams params;
  params.cuda = true;
  params.world_name = options.world_name;
  Scene scene = build.BuildScene(params);
  
  // I have little knowledge of setting the optimal block size.
  // 16 works for my PC.
  int bsize = 16;
  int gw = (size.width + bsize - 1) / bsize;
  int gh = (size.height + bsize - 1) / bsize;
  dim3 block_size(bsize, bsize);
  dim3 grid_size(gw, gh);
  
  EntityNodeBT<World> *world_node = CHECK_NOTNULL(build.SearchEntityByName<World>(options.world_name));
  auto *world = world_node->Get();
  
  //auto *integrator = world->GetIntegrator(); 
  auto *developer = CHECK_NOTNULL(world->data_host_.developer);
  auto *cu_developer = CHECK_NOTNULL(world->data_device_.developer);
  
  InitDeveloper<<<1, 1>>>(cu_developer, size);
  std::vector<AAFilter> aa_filters = GenerateSSAAFilters(options.aa);
  for (int i = 0; i < aa_filters.size(); ++i) {
    CHECK_NOTNULL(film.Data());
    CHECK_NOTNULL(cu_film_data_.get());
    GPUKernel<<<grid_size, block_size>>>(cu_film, scene, aa_filters[i]);
    Develop<<<1, 1>>>(cu_developer, cu_film, aa_filters[i].w); 
  }
  FinishDeveloper<<<1, 1>>>(cu_developer);
  cudaDeviceSynchronize();
  developer->Init(size);
  developer->RetrieveFromDevice(cu_developer);
  developer->Finish();
  return developer;
}

#endif

struct CPUImpl {
  
  CPUImpl(void) {}
  ~CPUImpl(void) {}
  
  Developer* Render(SceneBuilder &build, const RenderOptions &options);
};

Developer* CPUImpl::Render(
    SceneBuilder &build, const RenderOptions &options) {
  Size size = options.size;
  BuildSceneParams params;
  params.cuda = false;
  params.world_name = options.world_name;
  Scene scene = build.BuildScene(params);
  
  const Camera *camera = scene.GetCamera();
  Film film(size);
  
  int num_threads = options.num_threads;
  if (num_threads < 0) {num_threads = std::thread::hardware_concurrency();}
  
  std::vector<AAFilter> aa_filters = GenerateSSAAFilters(options.aa);
  
  auto *world_node = CHECK_NOTNULL(build.SearchEntityByName<World>(options.world_name));
  auto *world = world_node->Get();
  auto *integrator = world->data_host_.integrator;
  auto *developer = world->data_host_.developer;
  developer->Init(size);
  
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
          sample = integrator->Li(ray, scene);
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
    developer->Develop(film, aa.w);
  }
  developer->Finish();
  return developer;
}

class RTEngine::Impl {
 public:
  Impl(void);
  
  Developer* Render(SceneBuilder &build);
#ifdef WITH_CUDA  
  CUDAImpl cuda_impl;
#endif
  CPUImpl cpu_impl;
  
  RenderOptions options_;
};


RTEngine::Impl::Impl(void) {
}

void CheckIntegratorAndDeveloper(SceneBuilder *build, std::string world_name) {
  auto *world_node = CHECK_NOTNULL(build->SearchEntityByName<World>(world_name));
  auto *world = world_node->Get();
  auto *integrator = world->GetIntegrator();
  if (integrator == nullptr) {
    LOG(WARNING) << "The scene description does not specify a integrator. Use the default one.";
    build->CreateEntity<Integrator>("DefaultIntegrator", "_integrator");
    world->Parse({"SetIntegrator", "_integrator"}, build);
  }
  auto *developer = world->GetDeveloper();
  if (developer == nullptr) {
    LOG(WARNING) << "The scene description does not specify a developer. Use the default one.";
    build->CreateEntity<Developer>("DefaultDeveloper", "_developer");
    world->Parse({"SetDeveloper", "_developer"}, build);
  }
}

Developer* RTEngine::Impl::Render(
    SceneBuilder &build) {
  CheckIntegratorAndDeveloper(&build, options_.world_name);
  Developer *dev;
  if (options_.cuda) {
#ifdef WITH_CUDA
    dev = cuda_impl.Render(build, options_);
#else
    LOG(FATAL) << "qjulia2 is not compiled with CUDA support.";
#endif
  } else {
    dev = cpu_impl.Render(build, options_);
  }
  return dev;
}

RTEngine::RTEngine(void) : impl_(new Impl()) {
  
}

RTEngine::~RTEngine(void) {
}

void RTEngine::SetResolution(Size size) {
  impl_->options_.size = size;
}

void RTEngine::SetAAOption(AAOption aa) {
  impl_->options_.aa = aa;
}

void RTEngine::SetCUDA(bool enable) {
  impl_->options_.cuda = enable;
}

Developer* RTEngine::Render(
    SceneBuilder &build) {
  Timer timer;
  timer.Start();
  Developer *dev = impl_->Render(build);
  last_render_time_ = timer.End();
  return dev;
}

}

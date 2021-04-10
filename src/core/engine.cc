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

namespace qjulia {

struct RenderOptions {
  AAOption aa = AAOption::kSSAA6x;
  int num_threads = -1;
  std::string world_name = "";
  Size size;
};

class EngineCommon : public Engine {
 public:
   
  EngineCommon(void);
  
  SceneBuilder& GetSceneBuilder(void) override;
   
  void SetResolution(Size size) override;
  
  void SetAAOption(AAOption aa) override;
  
  void Parse(QJSDescription &descr) override;
  
  Float LastRenderTime(void) const override {return last_render_time_;} 
  
  RenderOptions options_;
  
  SceneBuilder scene_build_;
  
  Float last_render_time_ = 0;
};

EngineCommon::EngineCommon(void) {
  RegisterDefaultEntities(scene_build_);
}

void EngineCommon::SetResolution(Size size) {
  options_.size = size;
}

void EngineCommon::SetAAOption(AAOption aa) {
  options_.aa = aa;
}

SceneBuilder& EngineCommon::GetSceneBuilder(void) {
  return scene_build_;
}

void EngineCommon::Parse(QJSDescription &descr) {
  
  scene_build_.ParseSceneDescr(descr.scene);
  
  for (const auto &block : descr.engine.blocks) {
    if (block.name == "Engine") {
    } else if (block.name == "Developer") {
      for (const auto &statement : block.statements) {
        GetDeveloper().Parse(statement);
      }
    } else if (block.name == "DOFSimulator") {
      for (const auto &statement : block.statements) {
        GetDeveloper().GetDOFSimulator().Parse(statement);
      }
    } else {
      LOG(FATAL) << "Unknown engine object: " << block.name;
    }
  }
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
}

#ifdef WITH_CUDA

struct CUDAEngineImpl : public EngineCommon {
  
  CUDAEngineImpl(void);
  ~CUDAEngineImpl(void) {}
  
  void Render(void) override;
  
  Developer& GetDeveloper(void) override {return developer_;}
  
  DeveloperGPU developer_;
  int cu_film_data_size_ = 0;
  
  std::unique_ptr<Sample, void(*)(Sample*)> cu_film_data_ {
    nullptr, [](Sample *p) {cudaFree(p);}};
};

CUDAEngineImpl::CUDAEngineImpl(void) {
}

//CPU_AND_CUDA void MergeSample

KERNEL void GPUKernel(SampleFrame film, Scene scene, AAFilter aa) {
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (!film.IsValidCoords(r, c)) {return;}
  int i = film.GetIndex(r, c);
  
  DefaultIntegrator default_integrator;
  Integrator *integrator = scene.GetIntegrator();
  if (integrator == nullptr) {integrator = &default_integrator;}
  
  Float fr = r + aa.offset[0];
  Float fc = c + aa.offset[1];
  auto p = GenerateCameraCoords({fr, fc}, film.ArraySize());
  Ray ray = scene.GetCamera()->CastRay(p);
  film(i) = integrator->Li(ray, scene);
}

void CUDAEngineImpl::Render(void) {
  Timer timer;
  timer.Start();
  Size size = options_.size;
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
  SampleFrame cu_film(cu_film_data_.get(), size);  
  CHECK(!cu_film.HasOwnership());
  
  // Scene is only a simple structure that contains
  // the pointers to the world and the camera entities.
  // So we can pass the object to CUDA kernel as a parameter,
  // as long as the entities pointers are for the device.
  BuildSceneParams params;
  params.cuda = true;
  params.world_name = options_.world_name;
  Scene scene = scene_build_.BuildScene(params);
  
  // I have little knowledge of setting the optimal block size.
  // 16 works for my PC.
  int bsize = 16;
  int gw = (size.width + bsize - 1) / bsize;
  int gh = (size.height + bsize - 1) / bsize;
  dim3 block_size(bsize, bsize);
  dim3 grid_size(gw, gh);
  
  EntityNodeBT<World> *world_node = CHECK_NOTNULL(
    scene_build_.SearchEntityByName<World>(options_.world_name));
  auto *world = world_node->Get();
  
  //auto *integrator = world->GetIntegrator(); 
  //auto *developer = CHECK_NOTNULL(world->data_host_.developer);
  //auto *cu_developer = CHECK_NOTNULL(world->data_device_.developer);
  developer_.Init(size);
  std::vector<AAFilter> aa_filters = GenerateSSAAFilters(options_.aa);
  for (int i = 0; i < aa_filters.size(); ++i) {
    CHECK_NOTNULL(cu_film_data_.get());
    GPUKernel<<<grid_size, block_size>>>(cu_film, scene, aa_filters[i]);
    developer_.ProcessSampleFrame(cu_film, aa_filters[i].w);
  }
  developer_.Finish();
  cudaDeviceSynchronize();
  last_render_time_ = timer.End();
}

#endif

struct CPUEngineImpl : public EngineCommon {
  
  CPUEngineImpl(void) {}
  ~CPUEngineImpl(void) {}
  
  Developer& GetDeveloper(void) override {return developer_;}
  
  void Render(void) override;
  
  DeveloperCPU developer_;
};

void CPUEngineImpl::Render(void) {
  Timer timer;
  timer.Start();
  Size size = options_.size;
  BuildSceneParams params;
  params.cuda = false;
  params.world_name = options_.world_name;
  Scene scene = scene_build_.BuildScene(params);
  
  const Camera *camera = scene.GetCamera();
  SampleFrame film(size);
  
  int num_threads = options_.num_threads;
  if (num_threads < 0) {num_threads = std::thread::hardware_concurrency();}
  
  std::vector<AAFilter> aa_filters = GenerateSSAAFilters(options_.aa);
  auto *world_node = CHECK_NOTNULL(scene_build_.SearchEntityByName<World>(options_.world_name));
  auto *world = world_node->Get();
  auto *integrator = world->data_host_.integrator;
  developer_.Init(size);
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
          Float fr, fc;
          fr = r + aa.offset[0];
          fc = c + aa.offset[1];
          Vector2f p = GenerateCameraCoords({fr, fc}, film.ArraySize());
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
    developer_.ProcessSampleFrame(film, aa.w);
  }
  developer_.Finish();
  last_render_time_ = timer.End();
}

std::unique_ptr<Engine> CreateCPUEngine(void) {
  auto engine = std::make_unique<CPUEngineImpl>();
  return engine;
}

std::unique_ptr<Engine> CreateCUDAEngine(void) {
#ifdef WITH_CUDA
  auto engine = std::make_unique<CUDAEngineImpl>();
  return engine;
#else
  return {};
#endif
  
}

}

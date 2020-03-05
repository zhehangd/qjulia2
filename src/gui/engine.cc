#include <cmath>

#include "engine.h"

#include "core/qjulia2.h"

using namespace qjulia;

void Register(ResourceMgr &mgr) {
  mgr.RegisterPrototypeEntity(new Object());
  mgr.RegisterPrototypeEntity(new Scene());
  mgr.RegisterPrototypeEntity(new Material());
  mgr.RegisterPrototypeEntity(new Transform());
  mgr.RegisterPrototypeEntity(new SunLight());
  mgr.RegisterPrototypeEntity(new PointLight());
  mgr.RegisterPrototypeEntity(new OrthoCamera());
  mgr.RegisterPrototypeEntity(new PerspectiveCamera());
  mgr.RegisterPrototypeEntity(new Julia3DShape());
  mgr.RegisterPrototypeEntity(new SphereShape());
  mgr.RegisterPrototypeEntity(new PlaneShape());
}

void RenderEngine::Init(std::string scene_file) {
  
  size_ = cv::Size(640, 480);
  preview_size_ = cv::Size(64, 48);
  cache_.create(size_, CV_8UC3);
  prev_cache_.create(size_, CV_8UC3);
  
  Register(mgr_);
  mgr_.LoadSceneDescription(scene_file);
  
  Scene *scene = mgr_.GetScene();
  if (scene == nullptr) {
    std::cerr << "Error: Scene not found." << std::endl;
    return;
  }
  scene->SetActiveCamera(0);
}
  
void RenderEngine::SetValue(float v) {
  value_ = v;
}

cv::Size RenderEngine::GetSize(void) const {
  return size_;
}

cv::Mat RenderEngine::Render(void) {
  Run(size_, cache_);
  return cache_;
}

cv::Mat RenderEngine::Preview(void) {
  Run(preview_size_, prev_cache_);
  return prev_cache_;
}

void RenderEngine::Run(cv::Size size, cv::Mat &dst_image) {
  Options option;
  option.width = size.width;
  option.height = size.height;
  option.antialias = true;
  
  float angle = value_ / 99.0 * 3.1416;
  float dist = 5.3;
  float z = dist * std::cos(angle);
  float x = dist * std::sin(angle);
  
  Scene *scene = mgr_.GetScene();
  scene->SetActiveCamera(0);
  auto *camera = static_cast<PerspectiveCamera*>(const_cast<Camera*>(scene->GetActiveCamera()));
  camera->LookAt({x,1.2,z}, {0, 1.4, 0}, {0, 1, 0});
  
  DefaultIntegrator integrator;
  
  int num_threads = -1;
  
  Film film;
  RTEngine engine;
  engine.SetNumThreads(num_threads);
  engine.Render(*scene, integrator, option, &film);
  
  cv::Mat cache_small(option.height, option.width, CV_8UC3);
  for (int r = 0; r < option.height; ++r) {
    for (int c = 0; c < option.width; ++c) {
      const auto &sp = film.At(r, c).spectrum;
      float scale = 255;
      auto &dst = cache_small.at<cv::Vec3b>(r, c);
      for (int k = 0; k < 3; ++k) {
        dst[2 - k] = std::min(255, std::max(0, (int)std::round(sp[k] * scale)));
      }
    }
  }
  cv::resize(cache_small, dst_image, size_);
}

std::unique_ptr<RenderEngine> CreateDefaultEngine(void) {
  return std::make_unique<RenderEngine>();
}

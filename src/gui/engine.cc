#include <cmath>

#include "engine.h"

#include "core/qjulia2.h"
#include "core/camera/camera3d.h"

using namespace qjulia;

void RenderEngine::Init(std::string scene_file) {
  
  size_ = cv::Size(640, 480);
  preview_size_ = cv::Size(64, 48);
  cache_.create(size_, CV_8UC3);
  prev_cache_.create(size_, CV_8UC3);
  
  RegisterDefaultEntities(build);
  SceneDescr scene_descr = LoadSceneFile(scene_file);
  build.ParseSceneDescr(scene_descr);
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
  
  auto *camera = static_cast<Camera3D*>(ParseEntity<Camera>("", &build));
  
  float angle = value_ / 99.0 * 3.1416;
  float dist = 5.3;
  float z = dist * std::cos(angle);
  float x = dist * std::sin(angle);
  camera->LookAt({x,1.2,z}, {0, 1.4, 0}, {0, 1, 0});
  
  Film film(size.width, size.height);
  RenderOptions options;
  options.cuda = true;
  options.antialias = true;
  
  RTEngine engine;
  engine.Render(build, options, film);
  LOG(INFO) << "Rendering time: " << engine.LastRenderTime();
  
  cv::Mat cache_small(size.height, size.width, CV_8UC3);
  for (int r = 0; r < size.height; ++r) {
    for (int c = 0; c < size.width; ++c) {
      const auto &sp = film.At(r, c);
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

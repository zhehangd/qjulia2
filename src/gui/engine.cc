#include <cmath>

#include "engine.h"

#include "core/qjulia2.h"
#include "core/camera/camera3d.h"
#include "core/shape/julia3d.h"

using namespace qjulia;

void GUIRenderEngine::Init(std::string scene_file) {
  
  size_ = cv::Size(1280, 960);
  preview_size_ = cv::Size(160, 120);
  cache_.create(size_, CV_8UC3);
  prev_cache_.create(size_, CV_8UC3);
  
  RegisterDefaultEntities(build);
  SceneDescr scene_descr = LoadSceneFile(scene_file);
  build.ParseSceneDescr(scene_descr);
}
  
cv::Size GUIRenderEngine::GetSize(void) const {
  return size_;
}

cv::Mat GUIRenderEngine::Render(SceneOptions options) {
  Run(size_, cache_, options);
  return cache_;
}

cv::Mat GUIRenderEngine::Preview(SceneOptions options) {
  Run(preview_size_, prev_cache_, options);
  return prev_cache_;
}

void GUIRenderEngine::Run(cv::Size size, cv::Mat &dst_image, SceneOptions sopts) {
  
  auto *camera = static_cast<Camera3D*>(ParseEntity<Camera>("", &build));
  
  float dist = sopts.camera_pose[2];
  float azi = sopts.camera_pose[0] / 180.0f * 3.1416f;
  float alt = sopts.camera_pose[1] / 180.0f * 3.1416f;
  float y = dist * std::sin(alt) + 1.2;
  float z = dist * std::cos(alt) * std::cos(azi);
  float x = dist * std::cos(alt) * std::sin(azi);
  Vector3f camera_from(x, y, z);
  camera->LookAt(camera_from, {0, 1.4, 0}, {0, 1, 0});
  
  Quaternion jconst (sopts.julia_constant[0], sopts.julia_constant[1],
                     sopts.julia_constant[2], sopts.julia_constant[3]);
  
  //LOG(INFO) << camera_from << " " << jconst;
  
  auto *julia3d = static_cast<Julia3DShape*>(ParseEntity<Shape>("fractal_shape_1", &build));
  julia3d->SetConstant(jconst);
  
  Film film(size.width, size.height);
  RenderOptions options;
  options.cuda = true;
  options.antialias = true;
  
  engine.Render(build, options, film);
  LOG(INFO) << "time: " << engine.LastRenderTime();
  
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

std::unique_ptr<GUIRenderEngine> CreateDefaultEngine(void) {
  return std::make_unique<GUIRenderEngine>();
}

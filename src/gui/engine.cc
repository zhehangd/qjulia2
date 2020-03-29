#include <cmath>

#include "engine.h"

#include "core/qjulia2.h"
#include "core/camera/camera3d.h"
#include "core/shape/julia3d.h"

using namespace qjulia;

class RenderEngine::Impl {
 public:
  
  void Init(std::string scene_file);
  
  void Run(bool preview, cv::Mat &dst, SceneOptions options);
   
  cv::Size size_;
  cv::Size preview_size_;
  cv::Mat cache_;
  cv::Mat prev_cache_;
  
  qjulia::RTEngine engine;
  qjulia::SceneBuilder build;
};

void RenderEngine::Impl::Init(std::string scene_file) {
  
  size_ = cv::Size(1280, 960);
  preview_size_ = cv::Size(1280, 960);
  cache_.create(size_, CV_8UC3);
  prev_cache_.create(size_, CV_8UC3);
  
  RegisterDefaultEntities(build);
  SceneDescr scene_descr = LoadSceneFile(scene_file);
  build.ParseSceneDescr(scene_descr);
}

void RenderEngine::Impl::Run(bool preview, cv::Mat &dst_image, SceneOptions sopts) {
  
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
  
  RenderOptions options;
  options.cuda = true;
  options.antialias = true;
  
  Size size;
  if (preview) {
    size.width = preview_size_.width;
    size.height = preview_size_.height;
    options.antialias = false;
  } else {
    size.width = size_.width;
    size.height = size_.height;
    options.antialias = true;
  }
  
  Film film(size.width, size.height);
  
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

RenderEngine::RenderEngine(void) : impl_(new Impl()) {}

RenderEngine::~RenderEngine(void) {}

void RenderEngine::Init(std::string scene_file) {
  impl_->Init(scene_file);
}

cv::Size RenderEngine::GetSize(void) const {
  return impl_->size_;
}

cv::Mat RenderEngine::Render(SceneOptions options) {
  impl_->Run(false, impl_->cache_, options);
  return impl_->cache_;
}

cv::Mat RenderEngine::Preview(SceneOptions options) {
  impl_->Run(true, impl_->prev_cache_, options);
  return impl_->prev_cache_;
}

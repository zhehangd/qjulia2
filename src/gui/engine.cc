#include <cmath>

#include "engine.h"

#include "core/qjulia2.h"
#include "core/camera/camera3d.h"
#include "core/shape/julia3d.h"

using namespace qjulia;

enum class RenderType {
  kPreview,
  kDisplay,
  kSave,
};

class RenderEngine::Impl {
 public:
  
  void Init(std::string scene_file);
  
  void Run(RenderType rtype, Image &dst, SceneOptions options);
  
  Julia3DShape* GetJulia3D(void);
  
  Camera3D* GetCamera3D(void);
  
  RenderEngine::SceneOptions GetDefaultOptions();
   
  Size size_;
  Size preview_size_;
  Size save_size_;
  Image cache_;
  Image prev_cache_;
  
  qjulia::RTEngine engine;
  qjulia::SceneBuilder build;
};

void RenderEngine::Impl::Init(std::string scene_file) {
  
  size_ = Size(1280, 960);
  preview_size_ = Size(1280, 960);
  save_size_ = Size(2560, 1920);
  
  cache_.Resize(size_);
  prev_cache_.Resize(size_);
  
  RegisterDefaultEntities(build);
  SceneDescr scene_descr = LoadSceneFile(scene_file);
  build.ParseSceneDescr(scene_descr);
  
}

Julia3DShape* RenderEngine::Impl::GetJulia3D(void) {
  auto *julia3d = dynamic_cast<Julia3DShape*>(
    ParseEntity<Shape>("fractal_shape_1", &build));
  CHECK_NOTNULL(julia3d);
  return julia3d;
}

Camera3D* RenderEngine::Impl::GetCamera3D(void) {
  auto *camera = dynamic_cast<Camera3D*>(
    ParseEntity<Camera>("", &build));
  CHECK_NOTNULL(camera);
  return camera;
}

RenderEngine::SceneOptions RenderEngine::Impl::GetDefaultOptions() {
  //auto *camera = GetCamera3D();
  auto *julia3d = GetJulia3D();
  Quaternion jconst = julia3d->GetConstant();
  RenderEngine::SceneOptions opts;
  opts.fractal_constant = jconst;
  opts.fractal_precision = julia3d->GetPrecision();
  opts.fractal_cross_section = julia3d->GetCrossSectionFlag();
  opts.fractal_uv_black = julia3d->GetUVBlack();
  opts.fractal_uv_white = julia3d->GetUVWhite();
  opts.camera_pose = {10, 0, 5.3};
  opts.camera_target = {0, 1.2, 0};
  opts.realtime_image_size = {640, 360};
  opts.realtime_fast_image_size = {640, 360};
  opts.offline_image_size = {1920, 1080};
  opts.offline_filename = "output.png";
  return opts;
}

void RenderEngine::Impl::Run(RenderType rtype, Image &dst_image, SceneOptions sopts) {
  Vector3f camera_target = sopts.camera_target;
  Vector3f camera_position;
  float dist = sopts.camera_pose[2];
  float azi = sopts.camera_pose[0] / 180.0f * 3.1416f;
  float alt = sopts.camera_pose[1] / 180.0f * 3.1416f;
  camera_position[1] = std::sin(alt);
  camera_position[2] = std::cos(alt) * std::cos(azi);
  camera_position[0] = std::cos(alt) * std::sin(azi);
  camera_position = camera_position * dist + camera_target;
  auto *camera = GetCamera3D();
  camera->LookAt(camera_position, camera_target, {0, 1, 0});
  
  auto *julia3d = GetJulia3D();
  julia3d->SetConstant(sopts.fractal_constant);
  julia3d->SetPrecision(sopts.fractal_precision);
  julia3d->SetCrossSectionFlag(sopts.fractal_cross_section);
  julia3d->SetUVBlack(sopts.fractal_uv_black);
  julia3d->SetUVWhite(sopts.fractal_uv_white);
  
  RenderOptions options;
  options.cuda = true;
  options.antialias = true;
  
  Size size;
  if (rtype == RenderType::kPreview) {
    size.width = preview_size_.width;
    size.height = preview_size_.height;
    options.antialias = false;
  } else if (rtype == RenderType::kDisplay) {
    size.width = size_.width;
    size.height = size_.height;
    options.antialias = true;
  } else if (rtype == RenderType::kSave) {
    size.width = save_size_.width;
    size.height = save_size_.height;
    options.antialias = true;
  } else {
    LOG(FATAL) << "Unknown rtype";
  }
  
  Film film(size.width, size.height);
  
  engine.Render(build, options, film);
  LOG(INFO) << "time: " << engine.LastRenderTime();
  
  Image image(film);
  
  if (rtype == RenderType::kPreview) {
    UpSample(image, dst_image, size_);
  } else {
    image.CopyTo(dst_image);
  }
}

RenderEngine::RenderEngine(void) : impl_(new Impl()) {}

RenderEngine::~RenderEngine(void) {}

void RenderEngine::Init(std::string scene_file) {
  impl_->Init(scene_file);
}

RenderEngine::SceneOptions RenderEngine::GetDefaultOptions() {
  return impl_->GetDefaultOptions();
}

Image* RenderEngine::Render(SceneOptions options) {
  impl_->Run(RenderType::kDisplay, impl_->cache_, options);
  return &impl_->cache_;
}

Image* RenderEngine::Preview(SceneOptions options) {
  impl_->Run(RenderType::kPreview, impl_->prev_cache_, options);
  return &impl_->prev_cache_;
}

void RenderEngine::Save(SceneOptions options) {
  Image image;
  impl_->Run(RenderType::kSave, image, options);
  WritePNGImage(options.offline_filename, image);
}

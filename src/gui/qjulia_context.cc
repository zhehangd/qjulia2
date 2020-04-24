#include <cmath>
#include <map>

#include "qjulia_context.h"

#include "core/qjulia2.h"

#include "core/material.h"
#include "core/object.h"
#include "core/scene.h"
#include "core/world.h"
#include "core/texture.h"
#include "core/transform.h"

#include "core/camera/camera3d.h"
#include "core/light/simple.h"
#include "core/shape/julia3d.h"
#include "core/shape/plane.h"
#include "core/shape/sphere.h"

#include "module_point_light.h"
#include "module_sun_light.h"
#include "module_camera.h"
#include "module_julia3d_shape.h"

using namespace qjulia;

enum class RenderType {
  kPreview,
  kDisplay,
  kSave,
};

class QJuliaContext::Impl {
 public:
  
  void Init(void);
  
  void RegisterEntities(void);
  
  void Run(RenderType rtype, Image &dst, SceneCtrlParams options);
  
  Julia3DShape* GetJulia3D(void);
  
  Camera* GetCamera(void);
  
  SceneCtrlParams GetDefaultOptions();
  
  BaseModule* NewControlWidgetForBaseType(int btype_id);
  
  BaseModule* NewControlWidgetForSpecificType(int stype_id);
   
  Image cache_;
  Image prev_cache_;
  
  qjulia::RTEngine engine;
  qjulia::SceneBuilder build;
    
  // Maps a stype_id to a function that creates the 
  // control widgets of the corresponding entity.
  std::map<int, BaseModule*(*)(void)> stype_control_widget_lut_;
  std::map<int, BaseModule*(*)(void)> btype_control_widget_lut_;
};

void InitSceneBuild(SceneBuilder &build) {
  std::string text = 
    "Shape.Julia3D fractal_shape {\n"
    "  SetConstant 0.12,0.68,0.08,-0.46\n"
    "  SetPrecision 1e-4\n"
    "  SetUVBlack 0.36\n"
    "  SetUVWhite 0.63\n"
    "}\n"
    "\n"
    "Material material_fractal {\n"
    "  SetDiffuse 1,1,1\n"
    "  SetReflection 0.2\n"
    "  SetSpecular 0.3\n"
    "}\n"
    "\n"
    "Object fractal {\n"
    "  SetShape fractal_shape\n"
    "  SetMaterial material_fractal\n"
    "}\n"
    "\n"
    "Light.Point lamp {\n"
    "  SetPosition 0.2,-0.2,0\n"
    "  SetIntensity 0.11,0.18,0.20\n"
    "}\n"
    "Light.Sun sun {\n"
    "  SetOrientation -2,-3.5,-2\n"
    "  SetIntensity 0.88,1.27,2.56\n"
    "}\n"
    "\n"
    "Camera.Perspective camera {\n"
    "  LookAt 3.3,-0.67,2.0  0.5,0,0  0,1,0\n"
    "  SetFocus 1.8\n"
    "}\n"
    "\n"
    "World scene {\n"
    "  AddCamera camera\n"
    "  AddObject fractal\n"
    "  AddLight sun\n"
    "  AddLight lamp\n"
    "}\n";
  SceneDescr scene_descr = LoadSceneFromString(text);
  build.ParseSceneDescr(scene_descr);
}

void QJuliaContext::Impl::Init(void) {
  RegisterEntities();
  InitSceneBuild(build);
}

void QJuliaContext::Impl::RegisterEntities(void) {
  
  auto fn_camera = [](void) -> BaseModule* {return new CameraModule();};
  btype_control_widget_lut_[EntityTrait<Camera>::btype_id] = fn_camera;
  btype_control_widget_lut_[EntityTrait<Light>::btype_id] = nullptr;
  btype_control_widget_lut_[EntityTrait<Material>::btype_id] = nullptr;
  btype_control_widget_lut_[EntityTrait<Object>::btype_id] = nullptr;
  btype_control_widget_lut_[EntityTrait<Shape>::btype_id] = nullptr;
  btype_control_widget_lut_[EntityTrait<Texture>::btype_id] = nullptr;
  btype_control_widget_lut_[EntityTrait<Transform>::btype_id] = nullptr;
  btype_control_widget_lut_[EntityTrait<World>::btype_id] = nullptr;
  
  const SceneBuilder::RegRecord *reg = 0;
  
  reg = build.Register<Object>("");
  stype_control_widget_lut_[reg->stype_id] = nullptr;
  
  reg = build.Register<Material>("");
  stype_control_widget_lut_[reg->stype_id] = nullptr;
  
  reg = build.Register<Texture>("");
  stype_control_widget_lut_[reg->stype_id] = nullptr;
  
  reg = build.Register<Transform>("");
  stype_control_widget_lut_[reg->stype_id] = nullptr;

  reg = build.Register<World>("");
  stype_control_widget_lut_[reg->stype_id] = nullptr;
  
  reg = build.Register<PerspectiveCamera>("Perspective");
  stype_control_widget_lut_[reg->stype_id] = nullptr;

  reg = build.Register<OrthoCamera>("Ortho");
  stype_control_widget_lut_[reg->stype_id] = nullptr;

  reg = build.Register<PointLight>("Point");
  stype_control_widget_lut_[reg->stype_id] =
    [](void) -> BaseModule* {return new PointLightModule();};
  
  reg = build.Register<SunLight>("Sun");
  stype_control_widget_lut_[reg->stype_id] =
    [](void) -> BaseModule* {return new SunLightModule();};

  reg = build.Register<Julia3DShape>("Julia3D");
  stype_control_widget_lut_[reg->stype_id] =
    [](void) -> BaseModule* {return new Julia3DShapeModule();};

  reg = build.Register<PlaneShape>("Plane");
  stype_control_widget_lut_[reg->stype_id] = nullptr;

  reg = build.Register<SphereShape>("Sphere");
  stype_control_widget_lut_[reg->stype_id] = nullptr;
}

BaseModule* QJuliaContext::Impl::NewControlWidgetForBaseType(int btype_id) {
  auto it = btype_control_widget_lut_.find(btype_id);
  if ((it != btype_control_widget_lut_.end()) && (it->second != nullptr)) {
    return it->second();
  } else {
    return nullptr;
  }
}

BaseModule* QJuliaContext::Impl::NewControlWidgetForSpecificType(int stype_id) {
  auto it = stype_control_widget_lut_.find(stype_id);
  if ((it != stype_control_widget_lut_.end()) && (it->second != nullptr)) {
    return it->second();
  } else {
    return nullptr;
  }
}

Julia3DShape* QJuliaContext::Impl::GetJulia3D(void) {
  auto *julia3d = dynamic_cast<Julia3DShape*>(
    ParseEntity<Shape>("fractal_shape", &build));
  CHECK_NOTNULL(julia3d);
  return julia3d;
}

Camera* QJuliaContext::Impl::GetCamera(void) {
  auto *camera = ParseEntity<Camera>("", &build);
  CHECK_NOTNULL(camera);
  return camera;
}

SceneCtrlParams QJuliaContext::Impl::GetDefaultOptions() {
  //auto *camera = GetCamera();
  auto *julia3d = GetJulia3D();
  Quaternion jconst = julia3d->GetConstant();
  SceneCtrlParams opts;
  opts.fractal_constant = jconst;
  opts.fractal_precision = julia3d->GetPrecision();
  opts.fractal_cross_section = julia3d->GetCrossSectionFlag();
  opts.fractal_uv_black = julia3d->GetUVBlack();
  opts.fractal_uv_white = julia3d->GetUVWhite();
  opts.camera_pose = {10, 0, 5.3};
  opts.camera_target = {0, 0, 0};
  opts.realtime_image_size = {1920, 1080};
  opts.realtime_fast_image_size = {640, 360};
  opts.offline_image_size = {3840, 2160};
  opts.offline_filename = "output.png";
  return opts;
}

void QJuliaContext::Impl::Run(RenderType rtype, Image &dst_image, SceneCtrlParams sopts) {  
  RenderOptions options;
  options.cuda = true;
  options.antialias = true;
  
  Size size;
  if (rtype == RenderType::kPreview) {
    size = sopts.realtime_fast_image_size;
    options.antialias = false;
  } else if (rtype == RenderType::kDisplay) {
    size = sopts.realtime_image_size;
    options.antialias = true;
  } else if (rtype == RenderType::kSave) {
    size = sopts.offline_image_size;
    options.antialias = true;
  } else {
    LOG(FATAL) << "Unknown rtype";
  }
  
  Image image(size.width, size.height);
  engine.Render(build, options, image);
  LOG(INFO) << "time: " << engine.LastRenderTime();
  
  if (rtype == RenderType::kPreview) {
    UpSample(image, dst_image, sopts.realtime_image_size);
  } else {
    image.CopyTo(dst_image);
  }
}

QJuliaContext::QJuliaContext(void) : impl_(new Impl()) {}

QJuliaContext::~QJuliaContext(void) {}

void QJuliaContext::Init(void) {
  impl_->Init();
}

SceneCtrlParams QJuliaContext::GetDefaultOptions() {
  return impl_->GetDefaultOptions();
}

Image* QJuliaContext::Render(SceneCtrlParams options) {
  impl_->Run(RenderType::kDisplay, impl_->cache_, options);
  return &impl_->cache_;
}

Image* QJuliaContext::Preview(SceneCtrlParams options) {
  impl_->Run(RenderType::kPreview, impl_->prev_cache_, options);
  return &impl_->prev_cache_;
}

void QJuliaContext::Save(SceneCtrlParams options) {
  Image image;
  impl_->Run(RenderType::kSave, image, options);
  WritePNGImage(options.offline_filename, image);
}

SceneBuilder* QJuliaContext::GetSceneBuilder(void) {
  return &impl_->build;
}

BaseModule* QJuliaContext::NewControlWidgetForBaseType(int btype_id) {
  return impl_->NewControlWidgetForBaseType(btype_id);
}

BaseModule* QJuliaContext::NewControlWidgetForSpecificType(int stype_id) {
  return impl_->NewControlWidgetForSpecificType(stype_id);
}
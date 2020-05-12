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
#include "core/integrator/default.h"

#include "module_point_light.h"
#include "module_sun_light.h"
#include "module_camera.h"
#include "module_julia3d_shape.h"

using namespace qjulia;

enum class RenderType {
  kFastPreview,
  kDisplay,
  kSave,
};

class QJuliaContext::Impl {
 public:
  
  void Init(void);
  
  void RegisterEntities(void);
  
  void Run(RenderType rtype, Image &dst, SceneCtrlParams options);
  
  SceneCtrlParams GetDefaultOptions();
  
  BaseModule* NewControlWidgetForBaseType(int btype_id);
  
  BaseModule* NewControlWidgetForSpecificType(int stype_id);
   
  Image cache_;
  Image prev_cache_;
  
  qjulia::Size preview_size_;
  
  qjulia::Size fast_preview_size_;
  
  qjulia::Size render_size_;
  
  std::unique_ptr<qjulia::Engine> engine_;
  
  // Maps a stype_id to a function that creates the 
  // control widgets of the corresponding entity.
  std::map<int, BaseModule*(*)(void)> stype_control_widget_lut_;
  std::map<int, BaseModule*(*)(void)> btype_control_widget_lut_;
};

void InitSceneBuild(Engine &engine) {
  std::string text1 = 
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
    "  SetCamera camera\n"
    "  AddObject fractal\n"
    "  AddLight sun\n"
    "  AddLight lamp\n"
    "}\n";
  QJSDescription qjs_descr = LoadQJSFromString(text1);
  engine.Parse(qjs_descr);
}

void QJuliaContext::Impl::Init(void) {
  engine_ = qjulia::CreateCUDAEngine();
  RegisterEntities();
  InitSceneBuild(*engine_);
}

void QJuliaContext::Impl::RegisterEntities(void) {
  auto fn_camera = [](void) -> BaseModule* {return new CameraModule();};
  btype_control_widget_lut_[EntityTrait<Integrator>::btype_id] = nullptr;
  btype_control_widget_lut_[EntityTrait<Camera>::btype_id] = fn_camera;
  btype_control_widget_lut_[EntityTrait<Light>::btype_id] = nullptr;
  btype_control_widget_lut_[EntityTrait<Material>::btype_id] = nullptr;
  btype_control_widget_lut_[EntityTrait<Object>::btype_id] = nullptr;
  btype_control_widget_lut_[EntityTrait<Shape>::btype_id] = nullptr;
  btype_control_widget_lut_[EntityTrait<Texture>::btype_id] = nullptr;
  btype_control_widget_lut_[EntityTrait<Transform>::btype_id] = nullptr;
  btype_control_widget_lut_[EntityTrait<World>::btype_id] = nullptr;
  
  SceneBuilder &build = engine_->GetSceneBuilder();
  auto &table = build.GetRegTable();
  for (auto &reg : table) {
    if (reg.stype_name == "Point") {
      stype_control_widget_lut_[reg.stype_id] =
        [](void) -> BaseModule* {return new PointLightModule();};
    } else if (reg.stype_name == "Sun") {
      stype_control_widget_lut_[reg.stype_id] =
        [](void) -> BaseModule* {return new SunLightModule();};
    } else if (reg.stype_name == "Julia3D") {
      stype_control_widget_lut_[reg.stype_id] =
        [](void) -> BaseModule* {return new Julia3DShapeModule();};
    } else {
      stype_control_widget_lut_[reg.stype_id] = nullptr;
    }
  }
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

SceneCtrlParams QJuliaContext::Impl::GetDefaultOptions() {
  SceneCtrlParams opts;
  opts.realtime_image_size = {1920, 1080};
  opts.realtime_fast_image_size = {640, 360};
  opts.offline_image_size = {3840, 2160};
  opts.offline_filename = "output.png";
  return opts;
}

void QJuliaContext::Impl::Run(RenderType rtype, Image &dst_image, SceneCtrlParams sopts) {
  Size size;
  AAOption aa;
  if (rtype == RenderType::kFastPreview) {
    size = sopts.realtime_fast_image_size;
    aa = AAOption::kOff;
  } else if (rtype == RenderType::kDisplay) {
    size = sopts.realtime_image_size;
    aa = AAOption::kSSAA6x;
  } else if (rtype == RenderType::kSave) {
    size = sopts.offline_image_size;
    aa = AAOption::kSSAA6x;
  } else {
    LOG(FATAL) << "Unknown rtype";
  }
  
  engine_->SetAAOption(aa);
  engine_->SetResolution(size);
  
  engine_->Render();
  Developer &developer = engine_->GetDeveloper();
  
  Image image;
  developer.ProduceImage(image);
  
  
  LOG(INFO) << "time: " << engine_->LastRenderTime();
  
  if (rtype == RenderType::kFastPreview) {
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

Image* QJuliaContext::FastPreview(SceneCtrlParams options) {
  impl_->Run(RenderType::kFastPreview, impl_->prev_cache_, options);
  return &impl_->prev_cache_;
}

void QJuliaContext::Save(SceneCtrlParams options) {
  Image image;
  impl_->Run(RenderType::kSave, image, options);
  Imwrite(options.offline_filename, image);
}

SceneBuilder* QJuliaContext::GetSceneBuilder(void) {
  return &impl_->engine_->GetSceneBuilder();
}

BaseModule* QJuliaContext::NewControlWidgetForBaseType(int btype_id) {
  return impl_->NewControlWidgetForBaseType(btype_id);
}

BaseModule* QJuliaContext::NewControlWidgetForSpecificType(int stype_id) {
  return impl_->NewControlWidgetForSpecificType(stype_id);
}

void QJuliaContext::SaveScene(QString filename) {
  QJSDescription descr;
  descr.scene = impl_->engine_->GetSceneBuilder().SaveSceneDescr();
  SaveQJSToFile(filename.toStdString(), descr);
}

void QJuliaContext::SetFastPreviewSize(qjulia::Size size) {
  impl_->fast_preview_size_ = size;
}

void QJuliaContext::SetPreviewSize(qjulia::Size size) {
  impl_->preview_size_ = size;
}

void QJuliaContext::SetRenderSize(qjulia::Size size) {
  impl_->render_size_ = size;
}

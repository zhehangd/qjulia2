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
  
  RegisterDefaultEntities(build);
  
  SceneDescr scene_descr = LoadSceneFile(scene_file);
  for (int i = 0; i < (int)scene_descr.entities.size(); ++i) {
    LOG(INFO) << "Block #" << i << ":\n" << EntityDescr2Str(scene_descr.entities[i]) << "\n";
    const auto &edescr = scene_descr.entities[i];
    EntityNode *node = build.CreateEntity(edescr.type, edescr.subtype, edescr.name);
    CHECK_NOTNULL(node);
    Entity *e = node->Get();
    CHECK_NOTNULL(node);
    for (const auto &statement : edescr.statements) {
      e->Parse(statement, &build);
    }
  }
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
  
  auto *world = ParseEntityNode<World>("scene1", &build)->Get();
  auto *camera = ParseEntityNode<Camera>("camera_1", &build)->Get();
  
  float angle = value_ / 99.0 * 3.1416;
  float dist = 5.3;
  float z = dist * std::cos(angle);
  float x = dist * std::sin(angle);
  camera->LookAt({x,1.2,z}, {0, 1.4, 0}, {0, 1, 0});
  
  Scene scene;
  scene.camera_ = camera;
  scene.world_ = world;
  
  Film film;
  DefaultIntegrator integrator;
  
  Options option;
  option.width = size.width;
  option.height = size.height;
  option.antialias = true;
  
  RTEngine engine;
  engine.SetNumThreads(-1);
  engine.Render(scene, integrator, option, &film);
  LOG(INFO) << "Rendering time: " << engine.LastRenderTime();
  SaveToPPM(output_file, film, 255);
  
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

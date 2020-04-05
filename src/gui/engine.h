#ifndef QJULIA_GUI_ENGINE_H_
#define QJULIA_GUI_ENGINE_H_

#include <memory>
#include <string>

#include "core/qjulia2.h"

class RenderEngine {
 public:
  
  struct SceneOptions {
    qjulia::Quaternion fractal_constant;
    float fractal_precision;
    bool fractal_cross_section;
    float fractal_uv_black;
    float fractal_uv_white;
    
    qjulia::Size realtime_image_size;
    qjulia::Size realtime_fast_image_size;
    qjulia::Size offline_image_size;
    std::string offline_filename;
    
    qjulia::Vector3f camera_target;
    qjulia::Vector3f camera_pose; // azimuth/altitude/distance
    float camera_fov;
    float camera_headlight_lumin;
  };
   
  RenderEngine(void);
  ~RenderEngine(void);
  
  void Init(std::string scene_file);
  
  /// @brief Gets a copy of options based on the scene file.
  SceneOptions GetDefaultOptions();
  
  qjulia::Image* Render(SceneOptions options);
  
  qjulia::Image* Preview(SceneOptions options);
  
  void Save(SceneOptions options);
  
 private:
   
  class Impl;
  std::unique_ptr<Impl> impl_;
};


#endif

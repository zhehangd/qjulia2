#ifndef QJULIA_GUI_ENGINE_H_
#define QJULIA_GUI_ENGINE_H_

#include <memory>
#include <string>

#include <opencv2/opencv.hpp>

#include "core/qjulia2.h"

class RenderEngine {
 public:
   
  struct SceneOptions {
    qjulia::Quaternion julia_constant;
    qjulia::Vector3f camera_pose {10, 0, 5.3}; // azimuth/altitude/distance
    float precision;
    bool cross_section;
  };
   
  RenderEngine(void);
  ~RenderEngine(void);
  
  void Init(std::string scene_file);
  
  /// @brief Gets a copy of options based on the scene file.
  SceneOptions GetDefaultOptions();
  
  cv::Size GetSize(void) const;
  
  cv::Mat Render(SceneOptions options);
  
  cv::Mat Preview(SceneOptions options);
  
  void Save(SceneOptions options);
  
 private:
   
  class Impl;
  std::unique_ptr<Impl> impl_;
};


#endif

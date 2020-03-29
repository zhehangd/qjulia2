#ifndef QJULIA_GUI_ENGINE_H_
#define QJULIA_GUI_ENGINE_H_

#include <memory>
#include <string>

#include "backend_api.h"
#include "core/qjulia2.h"


class GUIRenderEngine : public RenderEngineInterface {
 public:
  
  void Init(std::string scene_file);
  
  cv::Size GetSize(void) const override;
  
  cv::Mat Render(SceneOptions options) override;
  
  cv::Mat Preview(SceneOptions options) override;
  
 private:
   
  void Run(cv::Size size, cv::Mat &dst, SceneOptions options);
   
  cv::Size size_;
  cv::Size preview_size_;
  cv::Mat cache_;
  cv::Mat prev_cache_;
  
  qjulia::RTEngine engine;
  qjulia::SceneBuilder build;
};

#endif

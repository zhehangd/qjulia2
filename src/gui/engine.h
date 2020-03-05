#ifndef QJULIA_GUI_ENGINE_H_
#define QJULIA_GUI_ENGINE_H_

#include <memory>
#include <string>

#include "backend_api.h"
#include "core/qjulia2.h"

class RenderEngine : public RenderEngineInterface {
 public:

  void Init(std::string scene_file);
  
  void SetValue(float v) override;
  
  cv::Size GetSize(void) const override;
  
  cv::Mat Render(void) override;
  
  cv::Mat Preview(void) override;
  
 private:
   
  void Run(cv::Size size, cv::Mat &dst);
   
  float value_ = 0;
  
  cv::Size size_;
  cv::Size preview_size_;
  cv::Mat cache_;
  cv::Mat prev_cache_;
  qjulia::ResourceMgr mgr_;
};

std::unique_ptr<RenderEngine> CreateDefaultEngine(void);

#endif

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
  
  cv::Mat& Render(void) override;
  
  cv::Mat& Preview(void) override;
  
 private:
   
  float value_ = 0;
  
  cv::Mat cache_;
  qjulia::ResourceMgr mgr_;
};

std::unique_ptr<RenderEngine> CreateDefaultEngine(void);

#endif

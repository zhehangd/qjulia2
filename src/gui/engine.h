#ifndef QJULIA_GUI_ENGINE_H_
#define QJULIA_GUI_ENGINE_H_

#include <memory>
#include <string>

#include "core/image.h"
#include "scene_ctrl_params.h"

class RenderEngine {
 public:
  
  RenderEngine(void);
  ~RenderEngine(void);
  
  void Init(std::string scene_file);
  
  /// @brief Gets a copy of options based on the scene file.
  SceneCtrlParams GetDefaultOptions();
  
  qjulia::Image* Render(SceneCtrlParams options);
  
  qjulia::Image* Preview(SceneCtrlParams options);
  
  void Save(SceneCtrlParams options);
  
 private:
   
  class Impl;
  std::unique_ptr<Impl> impl_;
};


#endif

#ifndef QJULIA_GUI_ENGINE_H_
#define QJULIA_GUI_ENGINE_H_

#include <memory>
#include <string>

#include <QString>

#include "core/image.h"
#include "core/scene_builder.h"
#include "scene_ctrl_params.h"
#include "module_base.h"

class QWidget;

class QJuliaContext {
 public:
  
  QJuliaContext(void);
  ~QJuliaContext(void);
  
  void Init(void);
  
  /// @brief Gets a copy of options based on the scene file.
  SceneCtrlParams GetDefaultOptions();
  
  qjulia::Image* Render(SceneCtrlParams options);
  
  qjulia::Image* Preview(SceneCtrlParams options);
  
  void Save(SceneCtrlParams options);
  
  qjulia::SceneBuilder* GetSceneBuilder(void);
  
  BaseModule* NewControlWidgetForBaseType(int btype_id);
  
  BaseModule* NewControlWidgetForSpecificType(int stype_id);
  
  void SaveScene(QString filename);
  
 private:
  
  class Impl;
  std::unique_ptr<Impl> impl_;
};


#endif

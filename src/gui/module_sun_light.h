#ifndef QJULIA_MODULE_SUN_LIGHT_H_
#define QJULIA_MODULE_SUN_LIGHT_H_

#include "module_base.h"

namespace Ui {
class SunLightModule;
}

namespace qjulia {
class SunLight;
}

class SunLightModule : public BaseModule {
  Q_OBJECT
 public:
  explicit SunLightModule(QWidget *parent = 0);
  ~SunLightModule();
  
  void AttachEntity(qjulia::Entity *e) override;
  
  void UpdateWidget(void) override;
  
  void UpdateEntity(void) override;
  
private:
  Ui::SunLightModule *ui;
  
  qjulia::SunLight *entity_;
};

#endif // QJULIA_MODULE_SUN_LIGHT_H_

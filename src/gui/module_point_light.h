#ifndef QJULIA_MODULE_POINT_LIGHT_H_
#define QJULIA_MODULE_POINT_LIGHT_H_

#include "module_base.h"

namespace Ui {
class PointLightModule;
}

namespace qjulia {
class PointLight;
}

class PointLightModule : public BaseModule {
  Q_OBJECT
 public:
  explicit PointLightModule(QWidget *parent = 0);
  ~PointLightModule();
  
  void AttachEntity(qjulia::Entity *e) override;
  
  void UpdateWidget(void) override;
  
  void UpdateEntity(void) override;
  
private:
  Ui::PointLightModule *ui;
  
  qjulia::PointLight *entity_;
};

#endif // QJULIA_MODULE_POINT_LIGHT_H_

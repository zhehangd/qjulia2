#ifndef QJULIA_CAMERA_MODULE_H_
#define QJULIA_CAMERA_MODULE_H_

#include "module_base.h"

namespace Ui {
class CameraModule;
}

namespace qjulia {
class Camera;
}

class CameraModule : public BaseModule {
  Q_OBJECT
 public:
  explicit CameraModule(QWidget *parent = 0);
  ~CameraModule();
  
  void AttachEntity(qjulia::Entity *e) override;
  
  void UpdateWidget(void) override;
  
  void UpdateEntity(void) override;

 private:
  Ui::CameraModule *ui;
  
  qjulia::Camera *entity_;
};

#endif // QJULIA_CAMERA_MODULE_H_

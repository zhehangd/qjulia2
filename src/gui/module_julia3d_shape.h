#ifndef QJULIA_MODULE_JULIA3D_SHAPE_H_
#define QJULIA_MODULE_JULIA3D_SHAPE_H_

#include "module_base.h"

namespace Ui {
class Julia3DShapeModule;
}

namespace qjulia {
class Julia3DShape;
}

class Julia3DShapeModule : public BaseModule {
  Q_OBJECT
 public:
  explicit Julia3DShapeModule(QWidget *parent = 0);
  ~Julia3DShapeModule();
  
  void AttachEntity(qjulia::Entity *e) override;
  
  void UpdateWidget(void) override;
  
  void UpdateEntity(void) override;
  
private:
  Ui::Julia3DShapeModule *ui;
  
  qjulia::Julia3DShape *entity_;
};

#endif // QJULIA_MODULE_JULIA3D_SHAPE_H_

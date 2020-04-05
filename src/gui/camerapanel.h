#ifndef CAMERAPANEL_H
#define CAMERAPANEL_H

#include <QWidget>

#include "panel.h"

namespace Ui {
class CameraPanel;
}

class CameraPanel : public Panel {
  Q_OBJECT

 public:
  explicit CameraPanel(QWidget *parent = 0);
  ~CameraPanel();
  
  void LinkToOptions(SceneCtrlParams *opts) override;

 private:
  
  void CollectParams(void) override;
  
  Ui::CameraPanel *ui;
};

#endif // CAMERAPANEL_H

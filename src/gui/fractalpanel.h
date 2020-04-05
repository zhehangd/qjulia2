#ifndef FRACTALPANEL_H
#define FRACTALPANEL_H

#include <QWidget>

#include "panel.h"

namespace Ui {
class FractalPanel;
}

class FractalPanel : public Panel {
  Q_OBJECT

 public:
  explicit FractalPanel(QWidget *parent = 0);
  ~FractalPanel();
  
  void LinkToOptions(SceneCtrlParams *opts) override;

 private:
  
  void CollectParams(void) override;
  
  Ui::FractalPanel *ui;
};

#endif // FRACTALPANEL_H

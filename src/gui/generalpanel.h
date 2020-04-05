#ifndef GENERALPANEL_H
#define GENERALPANEL_H

#include <QWidget>

#include "panel.h"

namespace Ui {
class GeneralPanel;
}

class GeneralPanel : public Panel {
  Q_OBJECT

 public:
  explicit GeneralPanel(QWidget *parent = 0);
  ~GeneralPanel();
  
  void LinkToOptions(SceneCtrlParams *opts) override;

 private:
  
  void CollectParams(void) override;
  
  Ui::GeneralPanel *ui;
  
 signals:
  void Save(void);

 protected slots:
   
  void onSave(bool);
};

#endif // GENERALPANEL_H

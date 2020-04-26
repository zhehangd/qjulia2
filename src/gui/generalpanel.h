#ifndef GENERALPANEL_H
#define GENERALPANEL_H

#include <QWidget>

#include "panel.h"
#include "qjulia_context.h"

namespace Ui {
class GeneralPanel;
}

class GeneralPanel : public Panel {
  Q_OBJECT

 public:
  explicit GeneralPanel(QWidget *parent = 0);
  ~GeneralPanel();
  
  void LinkToOptions(SceneCtrlParams *opts) override;
  
  void AttachContext(QJuliaContext *ctx);
  
 private:
  
  void CollectParams(void) override;
  
  Ui::GeneralPanel *ui;
  
  QJuliaContext *ctx_ = nullptr;
  
 signals:
  void Save(void);

 protected slots:
   
  void onSave(bool);
};

#endif // GENERALPANEL_H

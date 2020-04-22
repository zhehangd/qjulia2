#ifndef LIGHTCONTROL_H_
#define LIGHTCONTROL_H_

#include <QTimer>
#include <QWidget>

namespace Ui {
class LightControl;
}

class LightControl : public QWidget {
  Q_OBJECT
public:
  explicit LightControl(QWidget *parent = 0);
  ~LightControl();
  

private:
  Ui::LightControl *ui;
};


#endif // LIGHTCONTROL_H_

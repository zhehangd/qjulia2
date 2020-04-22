#ifndef COLORLCH_H_
#define COLORLCH_H_
#include <cmath>

#include <QWidget>

namespace Ui {
class ColorLCH;
}

#ifndef QJULIA_QT_PLUGIN
#include "core/vector.h"
#include "core/color.h"
#endif

class ColorLCH : public QWidget {
  Q_OBJECT
public:
  explicit ColorLCH(QWidget *parent = 0);
  ~ColorLCH();
  
#ifndef QJULIA_QT_PLUGIN
  qjulia::Vector3f GetLCHColor(void) const;
  void SetLCHColor(qjulia::Vector3f v);
  qjulia::Vector3f GetRGBColor(void) const;
  void SetRGBColor(qjulia::Vector3f v);
#endif

signals:
  void ValueChanging(void);
  void ValueChanged(void);

private:
  Ui::ColorLCH *ui;
};

#endif // COLORLCH_H_

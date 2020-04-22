#ifndef XYZCOORDS_H_
#define XYZCOORDS_H_

#include <QWidget>

namespace Ui {
class XyzCoords;
}

#ifndef QJULIA_QT_PLUGIN
#include "core/vector.h"
#endif

class XyzCoords : public QWidget {
  Q_OBJECT
public:
  explicit XyzCoords(QWidget *parent = 0);
  ~XyzCoords();
  
#ifndef QJULIA_QT_PLUGIN
  qjulia::Vector3f GetCoords(void) const;
  void SetCoords(qjulia::Vector3f v);
#endif

signals:
  void ValueChanging(void);
  void ValueChanged(void);

private:
  Ui::XyzCoords *ui;
};


#endif // XYZCOORDS_H_

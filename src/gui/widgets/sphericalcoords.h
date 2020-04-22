#ifndef SPHERICALCOORDS_H_
#define SPHERICALCOORDS_H_

#include <QWidget>

namespace Ui {
class SphericalCoords;
}

#ifndef QJULIA_QT_PLUGIN
#include "core/vector.h"
#endif

class SphericalCoords : public QWidget {
  Q_OBJECT
public:
  explicit SphericalCoords(QWidget *parent = 0);
  ~SphericalCoords();
  
#ifndef QJULIA_QT_PLUGIN
  qjulia::Vector3f GetSphericalCoords(void) const;
  void SetSphericalCoords(qjulia::Vector3f v);
  qjulia::Vector3f GetCartesianCoords(void) const;
  void SetCartesianCoords(qjulia::Vector3f v);
#endif

signals:
  void ValueChanging(void);
  void ValueChanged(void);

private:
  Ui::SphericalCoords *ui;
};


#endif // SPHERICALCOORDS_H_

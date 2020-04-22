#include "sphericalcoords.h"
#include "ui_sphericalcoords.h"

#ifndef QJULIA_QT_PLUGIN
#include "core/algorithm.h"
#endif

SphericalCoords::SphericalCoords(QWidget *parent) :
    QWidget(parent), ui(new Ui::SphericalCoords) {
  ui->setupUi(this);
  auto send_changing = [&]() {emit ValueChanging();};
  auto send_changed = [&]() {emit ValueChanged();};
  connect(ui->x, &ControlBar::valueChanging, send_changing);
  connect(ui->y, &ControlBar::valueChanging, send_changing);
  connect(ui->z, &ControlBar::valueChanging, send_changing);
  connect(ui->x, &ControlBar::valueChanged, send_changed);
  connect(ui->y, &ControlBar::valueChanged, send_changed);
  connect(ui->z, &ControlBar::valueChanged, send_changed);
}

SphericalCoords::~SphericalCoords() {
  delete ui;
}

#ifndef QJULIA_QT_PLUGIN
qjulia::Vector3f SphericalCoords::GetSphericalCoords(void) const {
  return {ui->x->GetValue(), ui->y->GetValue(), ui->z->GetValue()};
}

void SphericalCoords::SetSphericalCoords(qjulia::Vector3f v) {
  ui->x->SetValue(v[0]);
  ui->y->SetValue(v[1]);
  ui->z->SetValue(v[2]);
}

qjulia::Vector3f SphericalCoords::GetCartesianCoords(void) const {
  return Spherical2CartesianCoords(GetSphericalCoords());
}

void SphericalCoords::SetCartesianCoords(qjulia::Vector3f v) {
  SetSphericalCoords(Cartesian2SphericalCoords(v));
}
#endif

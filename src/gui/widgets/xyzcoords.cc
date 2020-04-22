#include "xyzcoords.h"
#include "ui_xyzcoords.h"

XyzCoords::XyzCoords(QWidget *parent)
    : QWidget(parent), ui(new Ui::XyzCoords) {
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

XyzCoords::~XyzCoords() {
  delete ui;
}

#ifndef QJULIA_QT_PLUGIN
qjulia::Vector3f XyzCoords::GetCoords(void) const {
  return {ui->x->GetValue(), ui->y->GetValue(), ui->z->GetValue()};
}

void XyzCoords::SetCoords(qjulia::Vector3f v) {
  ui->x->SetValue(v[0]);
  ui->y->SetValue(v[1]);
  ui->z->SetValue(v[2]);
}
#endif

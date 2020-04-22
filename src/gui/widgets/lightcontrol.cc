#include "lightcontrol.h"
#include "ui_lightcontrol.h"

#include <cmath>

#include <QDebug>

LightControl::LightControl(QWidget *parent) :
    QWidget(parent), ui(new Ui::LightControl) {
  ui->setupUi(this);
}

LightControl::~LightControl() {
  delete ui;
}

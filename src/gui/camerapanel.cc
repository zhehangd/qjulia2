#include "camerapanel.h"
#include "ui_camerapanel.h"

CameraPanel::CameraPanel(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::CameraPanel)
{
    ui->setupUi(this);
}

CameraPanel::~CameraPanel()
{
    delete ui;
}

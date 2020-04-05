#include "fractalpanel.h"
#include "ui_fractalpanel.h"

FractalPanel::FractalPanel(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::FractalPanel)
{
    ui->setupUi(this);
}

FractalPanel::~FractalPanel()
{
    delete ui;
}

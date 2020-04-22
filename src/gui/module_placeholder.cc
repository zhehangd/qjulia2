#include "module_placeholder.h"
#include "ui_module_placeholder.h"

PlaceholderModule::PlaceholderModule(QWidget *parent)
    : BaseModule(parent), ui(new Ui::PlaceholderModule) {
  ui->setupUi(this);
}

PlaceholderModule::~PlaceholderModule(void) {
  delete ui;
}

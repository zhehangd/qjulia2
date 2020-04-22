#ifndef QJULIA_MODULE_PLACEHOLDER_H_
#define QJULIA_MODULE_PLACEHOLDER_H_

#include "module_base.h"

namespace Ui {
class PlaceholderModule;
}

class PlaceholderModule : public BaseModule {
  Q_OBJECT
public:
  explicit PlaceholderModule(QWidget *parent = 0);
  ~PlaceholderModule();

private:
  Ui::PlaceholderModule *ui;
};

#endif // QJULIA_MODULE_PLACEHOLDER_H_

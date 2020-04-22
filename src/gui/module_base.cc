#include "module_base.h"

void BaseModule::OnValueChanging(void) {
  UpdateEntity();
  emit ValueChanging();
}

void BaseModule::OnValueChanged(void) {
  UpdateEntity();
  emit ValueChanged();
}

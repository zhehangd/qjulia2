#ifndef QJULIA_BASE_MODULE_H_
#define QJULIA_BASE_MODULE_H_

#include <QWidget>
#include <QDebug>

namespace qjulia {
class Entity;
}

class BaseModule : public QWidget {
  Q_OBJECT
 public:
  BaseModule(QWidget *parent = 0) : QWidget(parent) {}
  virtual ~BaseModule() {}
  
  /// @brief Make the object manipuate the given entity.
  /// 
  /// The entity must be convertible to the type the class
  /// is expecting. 
  virtual void AttachEntity(qjulia::Entity*) {}
  
  /// @brief Read the data from the entity and update the widget
  ///
  virtual void UpdateWidget(void) {}
  
  /// @brief Write the data to the entity.
  virtual void UpdateEntity(void) {}
  
signals:
  void ValueChanging(void);
  void ValueChanged(void);
  
protected slots:
  void OnValueChanging(void);
  void OnValueChanged(void);
};

#endif // QJULIA_BASE_MODULE_H_

#ifndef QJULIA_WIDGET_H_
#define QJULIA_WIDGET_H_

#include <QWidget>

/// @brief Base class for components
/// A component is a group of low-level widgets and can be used in many places,
/// such as a component to specify coordinates or colors.
class QJulia2Component : public QWidget {
 public:
  explicit QJulia2Component(QWidget *parent = 0) : QWidget(parent) {}
  
 signals:
  void InstantParamsChanging(void);
  void InstantParamsChanged(void);
  void LazyParamsChanged(void);
 
};


#endif

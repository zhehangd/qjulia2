#ifndef QJULIA_PANEL_H_
#define QJULIA_PANEL_H_

#include <QWidget>

#include "scene_ctrl_params.h"

class Panel : public QWidget {
  Q_OBJECT

 public:
  explicit Panel(QWidget *parent = 0) : QWidget(parent) {}
  virtual ~Panel() {}
  
  /// @brief Link a parameter struct
  ///
  /// This function has two effects. First, it causes the panel update
  /// the GUI according to the struct. Second, whenever the GUI signals
  /// a parameter update, the panel updates this struct to reflect
  /// the change.
  virtual void LinkToOptions(SceneCtrlParams *opts) {(void)opts;}

 private:
  
  // Update the parameters from GUI to the dst struct
  virtual void CollectParams(void) {}
 
 protected:
  
  SceneCtrlParams *opts_;

signals:
  void RealtimeParamsChanging(void);
  void RealtimeParamsChanged(void);

 protected slots:
   
  void onRealtimeParamsChanging(float) {onRealtimeParamsChanging();}
  void onRealtimeParamsChanged(float) {onRealtimeParamsChanged();}
  void onRealtimeParamsChanging(int) {onRealtimeParamsChanging();}
  void onRealtimeParamsChanged(int) {onRealtimeParamsChanged();}
  
  void onRealtimeParamsChanging(void);
  void onRealtimeParamsChanged(void);
  
};

inline void Panel::onRealtimeParamsChanging(void) {
  CollectParams();
  emit RealtimeParamsChanging();
}

inline void Panel::onRealtimeParamsChanged(void) {
  CollectParams();
  emit RealtimeParamsChanged();
}

#endif

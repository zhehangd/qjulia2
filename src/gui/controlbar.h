#ifndef CONTROLBAR_H_
#define CONTROLBAR_H_

#include <QTimer>
#include <QWidget>

namespace Ui {
class ControlBar;
}

class ControlBar : public QWidget {
  Q_OBJECT
  Q_PROPERTY(double valSrt READ GetStartValue WRITE SetStartValue)
  Q_PROPERTY(double valEnd READ GetEndValue WRITE SetEndValue)
  Q_PROPERTY(bool useLogScale READ GetLogScale WRITE SetLogScale)
  Q_PROPERTY(double value READ GetValue WRITE SetValue)

public:
  explicit ControlBar(QWidget *parent = 0);
  ~ControlBar();
  
  // This will not generate singals
  void SetValue(float value);
  
  float GetValue(void) const {return value_;}
  
  float GetStartValue(void) const {return vsrt_;}
  
  float GetEndValue(void) const {return vend_;}
  
  void SetStartValue(float from) {SetValueRange(from, vend_);}
  
  void SetEndValue(float to) {SetValueRange(vsrt_, to);}
  
  void SetValueRange(float from, float to);
  
  void SetLogScale(bool enable);
  
  bool GetLogScale(void) const {return use_log_;}
  
signals:
  void valueChanging(float);
  void valueChanged(float);

private:
  Ui::ControlBar *ui;
  
  // Decide whether value changing is completed.
  QTimer *value_changed_timer_;
  
  // Actual value maintained by this widget
  float value_ = 0;
  
  // Switch between linear/log scale for the slider
  bool use_log_ = false;
  
  float vsrt_ = 0;
  float vend_ = 1;
  int tsrt_ = 0;
  int tend_ = 100;
  
  // Convert slider position to actual value
  float TickToValue(int tick) const;
  
  // Convert value to slider position
  int ValueToTick(float value) const;
  
  // Start the timer of sending the valueChanged signal
  void TimeForValueChangedSignal(void);
  
  // Update the content of the text edit 
  void ShowEditContent(void);
  
  // Update the slider position
  void ShowSliderPosition(void);

private slots:
  void onSliderActed(int action);
  void onEditChanged(void);
  void SendValueChangedSignal(void) {emit valueChanged(value_);}
};


#endif // CONTROLBAR_H

#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <cmath>

#include <QMainWindow>
#include <QGraphicsPixmapItem>
#include <QFuture>
#include <QtConcurrent>
#include <QFutureWatcher>

#include "engine.h"

struct SliderCvt {
  virtual ~SliderCvt(void) {}
  virtual float TickToValue(int tick) const = 0;
  virtual int ValueToTick(float value) const = 0;
  float vsrt = 0;
  float vend = 1;
  int tsrt = 0;
  int tend = 100;
};

struct LinearSliderCvt : public SliderCvt {
  float TickToValue(int tick) const override {
    return (float)(tick - tsrt) / (tend - tsrt) * (vend - vsrt) + vsrt;}
  
  int ValueToTick(float value) const override {
    return (int)((value - vsrt) / (vend - vsrt) * (tend - tsrt) + tsrt);}
};

struct LogSliderCvt : public SliderCvt {
  float TickToValue(int tick) const override {
    float p = (float)(tick - tsrt) / (tend - tsrt);
    float v = vsrt * std::pow(vend / vsrt, p);
    return v;  
  }
  
  int ValueToTick(float value) const override {
    float k = std::log(value / vsrt) / std::log(vend / vsrt);
    int t = k * (tend - tsrt) + tsrt;
    return t;
  }
};

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow {
  Q_OBJECT

 public:
  explicit MainWindow(QWidget *parent, RenderEngine *engine);
  ~MainWindow();

 protected:
  void showEvent(QShowEvent *ev) override;
  
 private:
  
  void DrawImage(void);
  
  Ui::MainWindow *ui;
    
  QGraphicsPixmapItem pixmap_;
  
  QFutureWatcher<qjulia::Image*> render_watch_;
    
  RenderEngine *engine_;
  
  RenderEngine::SceneOptions engine_options_;
  
  LinearSliderCvt slider_azi_cvt_;
  LinearSliderCvt slider_alt_cvt_;
  LinearSliderCvt slider_dist_cvt_;
  LinearSliderCvt slider_jconst_cvt_;
  LogSliderCvt slider_precision_cvt_;
  LinearSliderCvt slider_uv_cvt_;

private slots:
  void onSliderAziChanged(int position);
  void onSliderAltChanged(int position);
  void onSliderDistChanged(int position);
  void onSliderJConst1Changed(int position);
  void onSliderJConst2Changed(int position);
  void onSliderJConst3Changed(int position);
  void onSliderJConst4Changed(int position);
  void onSliderPrecisionChanged(int position);
  void onSliderUV1Changed(int position);
  void onSliderUV2Changed(int position);
  void onPushButtonSaveClicked(bool checked);
  void onCheckBoxCrossSectionChanged(int state);
  
  // Render a full image
  void renderFull(void);
  
  // Called when engine finishes the rendering,
  // this functions will draw it on the screen.
  void onRenderFinished(void);
};

#endif // MAINWINDOW_H

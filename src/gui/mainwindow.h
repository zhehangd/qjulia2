#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QGraphicsPixmapItem>
#include <QFuture>
#include <QtConcurrent>
#include <QFutureWatcher>

#include <opencv2/opencv.hpp>

#include "backend_api.h"


struct SliderCvt {
  
  float TickToValue(int tick) {return (float)(tick - tsrt) / (tend - tsrt) * (vend - vsrt) + vsrt;}
  
  int ValueToTick(float value) {return (int)((value - vsrt) / (vend - vsrt) * (tend - tsrt) + tsrt);}
  
  float vsrt = 0;
  float vend = 1;
  int tsrt = 0;
  int tend = 99;
};

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow {
  Q_OBJECT

 public:
  explicit MainWindow(QWidget *parent, RenderEngineInterface *engine);
  ~MainWindow();

 protected:
  void showEvent(QShowEvent *ev) override;
  
 private:
  
  void DrawImage(int pos=0);
   
  Ui::MainWindow *ui;
    
  QGraphicsPixmapItem pixmap_;
  
  QFutureWatcher<cv::Mat> render_watch_;
    
  RenderEngineInterface *engine_;
  
  RenderEngineInterface::SceneOptions engine_options_;
  
  SliderCvt slider_azi_cvt_;
  SliderCvt slider_alt_cvt_;
  SliderCvt slider_dist_cvt_;
  SliderCvt slider_jconst_cvt_;

private slots:
  void onSliderAziChanged(int position);
  void onSliderAltChanged(int position);
  void onSliderDistChanged(int position);
  void onSliderJConst1Changed(int position);
  void onSliderJConst2Changed(int position);
  void onSliderJConst3Changed(int position);
  void onSliderJConst4Changed(int position);
  
  // Render a full image
  void renderFull(void);
  
  // Called when engine finishes the rendering,
  // this functions will draw it on the screen.
  void onRenderFinished(void);
};

#endif // MAINWINDOW_H

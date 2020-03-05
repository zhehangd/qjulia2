#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QGraphicsPixmapItem>
#include <QFuture>
#include <QtConcurrent>
#include <QFutureWatcher>

#include <opencv2/opencv.hpp>

#include "backend_api.h"

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
  
  int value_ = 0;

private slots:
  void onSliderValueChanged(int position);
  void onSliderReleased(void);
  void onRenderFinished(void);
};

#endif // MAINWINDOW_H

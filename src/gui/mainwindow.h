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
  
  RenderEngineInterface::SceneOptions engine_options_;
  
  // Send current settings to engine to render a preview
  void updatePreview(void);
  
  // Send current settings to engine to render a full
  void updateFull(void);
  
  int value_ = 0;

private slots:
  void onSliderAziChanged(int position);
  void onSliderAziReleased(void);
  void onSliderAltChanged(int position);
  void onSliderAltReleased(void);
  void onSliderDistChanged(int position);
  void onSliderDistReleased(void);
  void onRenderFinished(void);
};

#endif // MAINWINDOW_H

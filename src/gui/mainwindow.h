#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QGraphicsPixmapItem>
#include <QFuture>
#include <QtConcurrent>
#include <QFutureWatcher>

#include "qjulia_context.h"
#include "scene_ctrl_params.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow {
  Q_OBJECT

public:
  explicit MainWindow(QWidget *parent, QJuliaContext *engine);
  ~MainWindow();

private:
  Ui::MainWindow *ui;
    
  QGraphicsPixmapItem pixmap_;
  
  QFutureWatcher<qjulia::Image*> render_watch_;
    
  QJuliaContext *ctx_;
  
  SceneCtrlParams scene_params_;
  
  void DrawImage(void);
  
private slots:
  
  // Called when engine finishes the rendering,
  // this functions will draw it on the screen.
  void onRenderFinished(void);
  
  
  void onRealtimeParamsChanging(void);
  void onRealtimeParamsChanged(void);
  void onRenderAndSave(void);
  
};

#endif // MAINWINDOW_H

#ifndef FRACTALPANEL_H
#define FRACTALPANEL_H

#include <QWidget>

namespace Ui {
class FractalPanel;
}

class FractalPanel : public QWidget
{
    Q_OBJECT

public:
    explicit FractalPanel(QWidget *parent = 0);
    ~FractalPanel();

private:
    Ui::FractalPanel *ui;
};

#endif // FRACTALPANEL_H

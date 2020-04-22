#include "widgets/controlbar.h"
#include "controlbarplugin.h"

#include <QtPlugin>

ControlBarPlugin::ControlBarPlugin(QObject *parent)
    : QObject(parent)
{
    m_initialized = false;
}

void ControlBarPlugin::initialize(QDesignerFormEditorInterface * /* core */)
{
    if (m_initialized)
        return;

    // Add extension registrations, etc. here

    m_initialized = true;
}

bool ControlBarPlugin::isInitialized() const
{
    return m_initialized;
}

QWidget *ControlBarPlugin::createWidget(QWidget *parent)
{
    return new ControlBar(parent);
}

QString ControlBarPlugin::name() const
{
    return QLatin1String("ControlBar");
}

QString ControlBarPlugin::group() const
{
    return QLatin1String("");
}

QIcon ControlBarPlugin::icon() const
{
    return QIcon();
}

QString ControlBarPlugin::toolTip() const
{
    return QLatin1String("");
}

QString ControlBarPlugin::whatsThis() const
{
    return QLatin1String("");
}

bool ControlBarPlugin::isContainer() const
{
    return false;
}

QString ControlBarPlugin::domXml() const
{
    return QLatin1String("<widget class=\"ControlBar\" name=\"controlBar\">\n</widget>\n");
}

QString ControlBarPlugin::includeFile() const
{
    return QLatin1String("widgets/controlbar.h");
}
#if QT_VERSION < 0x050000
Q_EXPORT_PLUGIN2(controlbarplugin, ControlBarPlugin)
#endif // QT_VERSION < 0x050000

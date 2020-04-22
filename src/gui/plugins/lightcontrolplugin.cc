#include "widgets/lightcontrol.h"
#include "lightcontrolplugin.h"

#include <QtPlugin>

LightControlPlugin::LightControlPlugin(QObject *parent)
    : QObject(parent)
{
    m_initialized = false;
}

void LightControlPlugin::initialize(QDesignerFormEditorInterface * /* core */)
{
    if (m_initialized)
        return;

    // Add extension registrations, etc. here

    m_initialized = true;
}

bool LightControlPlugin::isInitialized() const
{
    return m_initialized;
}

QWidget *LightControlPlugin::createWidget(QWidget *parent)
{
    return new LightControl(parent);
}

QString LightControlPlugin::name() const
{
    return QLatin1String("LightControl");
}

QString LightControlPlugin::group() const
{
    return QLatin1String("");
}

QIcon LightControlPlugin::icon() const
{
    return QIcon();
}

QString LightControlPlugin::toolTip() const
{
    return QLatin1String("");
}

QString LightControlPlugin::whatsThis() const
{
    return QLatin1String("");
}

bool LightControlPlugin::isContainer() const
{
    return false;
}

QString LightControlPlugin::domXml() const
{
    return QLatin1String("<widget class=\"LightControl\" name=\"lightControl\">\n</widget>\n");
}

QString LightControlPlugin::includeFile() const
{
    return QLatin1String("widgets/lightcontrol.h");
}
#if QT_VERSION < 0x050000
Q_EXPORT_PLUGIN2(lightcontrolplugin, LightControlPlugin)
#endif // QT_VERSION < 0x050000

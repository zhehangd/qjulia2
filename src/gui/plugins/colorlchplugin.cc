#include "widgets/colorlch.h"
#include "colorlchplugin.h"

#include <QtPlugin>

ColorLCHPlugin::ColorLCHPlugin(QObject *parent)
    : QObject(parent)
{
    m_initialized = false;
}

void ColorLCHPlugin::initialize(QDesignerFormEditorInterface * /* core */)
{
    if (m_initialized)
        return;

    // Add extension registrations, etc. here

    m_initialized = true;
}

bool ColorLCHPlugin::isInitialized() const
{
    return m_initialized;
}

QWidget *ColorLCHPlugin::createWidget(QWidget *parent)
{
    return new ColorLCH(parent);
}

QString ColorLCHPlugin::name() const
{
    return QLatin1String("ColorLCH");
}

QString ColorLCHPlugin::group() const
{
    return QLatin1String("");
}

QIcon ColorLCHPlugin::icon() const
{
    return QIcon();
}

QString ColorLCHPlugin::toolTip() const
{
    return QLatin1String("");
}

QString ColorLCHPlugin::whatsThis() const
{
    return QLatin1String("");
}

bool ColorLCHPlugin::isContainer() const
{
    return false;
}

QString ColorLCHPlugin::domXml() const
{
    return QLatin1String("<widget class=\"ColorLCH\" name=\"colorLCH\">\n</widget>\n");
}

QString ColorLCHPlugin::includeFile() const
{
    return QLatin1String("widgets/colorlch.h");
}
#if QT_VERSION < 0x050000
Q_EXPORT_PLUGIN2(colorlchplugin, ColorLCHPlugin)
#endif // QT_VERSION < 0x050000

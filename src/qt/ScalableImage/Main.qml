import QtQuick
import QtQuick.Layouts
import QtQuick.Window

Window {
    width: 640
    height: 480
    visible: true
    title: qsTr("Hello World")

    QuickScalableImage {
        anchors.fill: parent
        Component.onCompleted: {
            source = "file:///F:/data/VOC/VOC2007/train/images/000012.jpg"
        }
    }
}

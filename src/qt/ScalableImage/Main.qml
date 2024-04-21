import QtQuick
import QtQuick.Layouts
import QtQuick.Window
import QtQuick.Controls

import ScalableImage as T

Window {
    width: 1920
    height: 1080
    visible: true
    title: qsTr("Hello World")

//     SplitView {
//         anchors.fill: parent
//         QuickScalableImage3 {
//             id: t
//             isFitInView: false
// //            anchors.fill: parent
//             SplitView.fillHeight: true
//             SplitView.fillWidth: true
//             Component.onCompleted: {
//                 source = "file:///D:/Datasets/Photos/raw/F3f6LKvW4AAgnDi.jpg"
//             }
//             onImageRectChanged: {
//                 if (t.imageRect.length <  4)
//                     return
//                 r.data = t.imageRect
//             }
//         }
//         Item {
//             id: b
//             implicitWidth: 500
//             SplitView.fillHeight: true
//             Image {
//                 id: _img
//                 clip: true
//                 anchors.fill: parent
//                 fillMode: Image.PreserveAspectFit
//                 property real scaleValue: Math.min(height / sourceSize.height, width / sourceSize.width)
//                 property real xOffset: Math.abs(width - paintedWidth) / 2
//                 property real yOffset: Math.abs(height - paintedHeight) / 2
//                 source: "file:///D:/Datasets/Photos/raw/F3f6LKvW4AAgnDi.jpg"
//                 Rectangle {
//                     id: r
//                     property var data: [0,0,0,0]
//                     x: _img.xOffset + data[0] * _img.scaleValue
//                     y: _img.yOffset + data[1] * _img.scaleValue
//                     width: data[2] * _img.scaleValue
//                     height: data[3] * _img.scaleValue
//                     color: "transparent"
//                     border.width: 5
//                     border.color: "red"
//                 }

//             }

//         }
//     }


    Rectangle {
        x: 200
        y: 200
        radius: 50
        width: 200
        height: 200
        color: "blue"
        border.width: 10
        border.color: "red"
    }

    T.QuickRectangle {
        visible: true
        x: 200
        y: 450
        radius: 50
        width: 200
        height: 200
        color: "yellow"
        border.width: 10
        border.color: "red"
        // border.style: Qt.DashLine
    }

    Rectangle {
        x: 450
        y: 200
        radius: 100
        width: 200
        height: 200
        color: "blue"
        border.width: 10
        border.color: "red"
    }

    T.QuickCircle {
        // x: 450
        // y: 450

        center.x: 550
        center.y: 550

        radius: 100
        color: "blue"
        border.width: 10
        border.color: "red"
        // border.style: Qt.DashLine
    }

}


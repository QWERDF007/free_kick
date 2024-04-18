import QtQuick
import QtQuick.Controls

import QtQuick.Shapes

Shape {
    id: control
    property real radius: 0
    property color color: "transparent"

    property int borderWidth: 0
    property color borderColor: control.color
    property int borderStyle: ShapePath.SolidLine

    ShapePath {
        startX: 0
        startY: 0
        fillColor: control.color
        strokeColor: control.borderColor
        strokeWidth: control.borderWidth
        strokeStyle: control.borderStyle
        PathQuad { x: control.radius; y: 0; controlX: 0; controlY: 0 }
        PathLine { x: control.width - control.radius; y: 0 }
        PathQuad { x: control.width; y: control.radius; controlX: control.width; controlY: 0 }
        PathLine { x: control.width; y: control.height - control.radius }
        PathQuad { x: control.width - control.radius; y: control.height; controlX: control.width; controlY: control.height }
        PathLine { x: control.radius; y: control.height }
        PathQuad { x: 0; y: control.height - control.radius; controlX: 0; controlY: control.height }
        PathLine { x: 0; y: control.radius }
    }
}

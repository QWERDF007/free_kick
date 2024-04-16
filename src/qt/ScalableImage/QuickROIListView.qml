import QtQuick
import QtQuick.Controls

Repeater {
    id: regions
    delegate: Rectangle {
        id: roi
        x: model.x
        y: model.y
        width: model.width
        height: model.height
        color: model.color
        property real originalOpacity: 0.3


        opacity: {
            if (roiMouseArea.containsMouse || roi.selected) {
                return originalOpacity * 2
            }
            return originalOpacity
        }

        onXChanged: {
            if (roiMouseArea.pressed && roi.selected) {
                model.x = Math.min(Math.max(0, x), imagePaintedRegion.width)
            }
        }

        onYChanged: {
            if (roiMouseArea.pressed && roi.selected) {
                model.y = Math.min(Math.max(0, y), imagePaintedRegion.height)
            }
        }

        onWidthChanged: {
            if (roiMouseArea.pressed && roi.selected) {
                model.width = Math.min(roi.width, imagePaintedRegion.width)
            }
        }

        onHeightChanged: {
            if (roiMouseArea.pressed && roi.selected) {
                model.height = Math.min(roi.height, imagePaintedRegion.height)
            }
        }

        MouseArea {
            id: roiMouseArea
            anchors.fill: parent
            anchors.leftMargin: -5
            anchors.rightMargin: -5
            anchors.topMargin: -5
            anchors.bottomMargin: -5
            hoverEnabled: true
            drag.target: roiMouseArea.pressed ? roi : null
            drag.axis: Drag.XAndYAxis
            drag.minimumX: 0
            drag.maximumX: imagePaintedRegion.width - roi.width
            drag.minimumY: 0
            drag.maximumY: imagePaintedRegion.height - roi.height

            onPressed: function(mouse) {
                console.log("roi onPressed roi index", index, model.index, mouse.x, mouse.y)
                ism.select(rois.index(model.index, 0), ItemSelectionModel.Select | ItemSelectionModel.Current)
            }

            onPositionChanged: function(mouse) {
                var pt = mapToItem(roi, mouse.x, mouse.y)
                //                            console.log("roi onPositionChanged", mouse.x, mouse.y, pt.x, pt.y, isNearCorner)
                if (imagePaintedRegion.isPointNear(pt.x, pt.y, 0, 0, 10)) {
                    setCursorShape(Qt.SizeFDiagCursor)
                } else if (imagePaintedRegion.isPointNear(pt.x, pt.y, imagePaintedRegion.width, imagePaintedRegion.height, 10)) {
                    setCursorShape(Qt.SizeFDiagCursor)
                } else {
                    setCursorShape(Qt.ArrowCursor)
                }
            }
        }

        function setCursorShape(cursorShape) {
            if (roiMouseArea.cursorShape !== cursorShape) {
                roiMouseArea.cursorShape = cursorShape
            }
        }
    }
}

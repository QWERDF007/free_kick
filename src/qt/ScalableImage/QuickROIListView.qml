import QtQuick
import QtQuick.Controls

Item {
    id: imagePaintedRegion
    width: 200
    height: 200

    property alias rois: regions.model
    property alias selection: ism
    property real _m: 20 / imagePaintedRegion.parent.scale

    ItemSelectionModel {
        id: ism
        model: rois
    }

    function isPointNear(px, py, cx, cy, radius) {
        var dx = px - cx
        var dy = py - cy
        var dist = Math.sqrt(dx * dx + dy * dy)
        return dist < radius
    }

    function getItemsUnderMouse(pos) {
        var items = []
        for (var i = 0; i < imagePaintedRegion.children.length; ++i) {
            var child = imagePaintedRegion.children[i]
            if (child instanceof Rectangle)  {
                var posOnChild = mapToItem(child, pos)
                if (child.visible && child.contains(posOnChild)) {
                    items.push(child)
                }
            }
        }
        return items
    }

    function selectOneItem(pos) {
        var items = imagePaintedRegion.getItemsUnderMouse(pos)
        if (items.length < 1)
            return
        var tempIndex = -1
        for (var i = 0; i < items.length; ++i) {
            if (regions.highlightIndex === items[i].index) {
                tempIndex = i
            }
        }
        --tempIndex
        regions.highlightIndex = tempIndex < 0 ? items[items.length - 1].index : items[tempIndex].index
        ism.select(rois.index(regions.highlightIndex, 0), ItemSelectionModel.Select | ItemSelectionModel.Current)
    }



    Repeater {
        id: regions
        // model: rois
        property int highlightIndex: -1
        delegate: Rectangle {
            id: roi
            x: model.x
            y: model.y
            width: model.width
            height: model.height
            color: model.color
            property real originalOpacity: 0.3
            property bool selected: ism.isRowSelected(model.index)
            property int index: model.index
            Component.onCompleted: {
                console.log("roi.parent", roi.parent)
            }

            Connections{
                target: ism
                function onSelectionChanged (selected, deselected) {
                    roi.selected = ism.isRowSelected(model.index)
                }
            }
            opacity: roi.selected ? originalOpacity * 2 :  originalOpacity

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
                anchors.margins: -5
                hoverEnabled: true
                drag.target: roiMouseArea.pressed && roi.selected ? roi : null
                drag.axis: Drag.XAndYAxis
                drag.minimumX: 0
                drag.maximumX: imagePaintedRegion.width - roi.width
                drag.minimumY: 0
                drag.maximumY: imagePaintedRegion.height - roi.height


                onPressed: function(mouse) {

                }

                onReleased: function(mouse) {
                    console.log("onReleased")
                    var pos = mapToItem(imagePaintedRegion, mouse.x, mouse.y)
                    imagePaintedRegion.selectOneItem(pos)
                }

                onPositionChanged: function(mouse) {
                    if (roi.selected) {
                        var pt = mapToItem(roi, mouse.x, mouse.y)
                        if (imagePaintedRegion.isPointNear(pt.x, pt.y, 0, 0, imagePaintedRegion._m)) {
                            setCursorShape(Qt.SizeFDiagCursor)
                        } else if (imagePaintedRegion.isPointNear(pt.x, pt.y, roi.width, roi.height, imagePaintedRegion._m)) {
                            setCursorShape(Qt.SizeFDiagCursor)
                        } else if (imagePaintedRegion.isPointNear(pt.x, pt.y, 0, roi.height, imagePaintedRegion._m)) {
                            setCursorShape(Qt.SizeBDiagCursor)
                        } else if (imagePaintedRegion.isPointNear(pt.x, pt.y, roi.width, 0, imagePaintedRegion._m)) {
                            setCursorShape(Qt.SizeBDiagCursor)
                        } else {
                            setCursorShape(Qt.ArrowCursor)
                        }
                    }
                }
            }

            Keys.onDeletePressed: {

            }

            function setCursorShape(cursorShape) {
                if (roiMouseArea.cursorShape !== cursorShape) {
                    roiMouseArea.cursorShape = cursorShape
                }
            }
        }
    }
}

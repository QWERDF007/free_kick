import QtQuick
import QtQuick.Controls

QuickRectangle {
    id: editableRect
    property bool selected: false
    property real _opacity: 0.3
    property real _m: 10 / parent.scale
    property int minimalSize: 5
    opacity: selected ? _opacity * 2 : _opacity
    // visible: false
//    border.color: "red"
//    border.width: selected ? 2 : 1
    borderColor: "red"
    borderWidth: selected ? 2 : 1
    color: "transparent"
    // color: "red"
    MouseArea {
        id: mouseArea
        property bool dragEnable: false
        property int dragType: -1
        anchors.fill: parent
        anchors.margins: -10
        acceptedButtons: Qt.AllButtons
        hoverEnabled: true
        drag.target: dragEnable ? editableRect : null
        drag.minimumX: 0
        drag.maximumX: _image.width - editableRect.width
        drag.minimumY: 0
        drag.maximumY: _image.height - editableRect.height
        onPressed: function(mouse) {
            if (mouse.button === Qt.LeftButton) {
                var pos = mapToItem(editableRect, mouse.x, mouse.y)
                if (editableRect.selected && pos.x > 0 && pos.x < editableRect.width && pos.y > 0 && pos.y < editableRect.height) {
                    dragEnable = true
                }
            } else if (mouse.button === Qt.MiddleButton) {
                mouse.accepted = false
            }
        }

        onReleased: function(mouse) {
            if (mouse.button === Qt.LeftButton) {
                dragEnable = false
                editableRect.selected = true
                var pt = mapToItem(editableRect, mouse.x, mouse.y)
                if (editableRect.isPointNearPoint(pt.x, pt.y, 0, 0, editableRect._m)) { // top left
                    editableRect.setCursorShape(Qt.SizeFDiagCursor)
                } else if (editableRect.isPointNearPoint(pt.x, pt.y, editableRect.width, editableRect.height, editableRect._m)) { // bottom right
                    editableRect.setCursorShape(Qt.SizeFDiagCursor)
                } else if (editableRect.isPointNearPoint(pt.x, pt.y, 0, editableRect.height, editableRect._m)) { // bottom left
                    editableRect.setCursorShape(Qt.SizeBDiagCursor)
                } else if (editableRect.isPointNearPoint(pt.x, pt.y, editableRect.width, 0, editableRect._m)) { // top right
                    editableRect.setCursorShape(Qt.SizeBDiagCursor)
                } else if (editableRect.isPointNearLine(pt.x, pt.y, 0, 0, 0, editableRect.height, editableRect._m)) { // left edge
                    editableRect.setCursorShape(Qt.SizeHorCursor)
                } else if (editableRect.isPointNearLine(pt.x, pt.y, editableRect.width, 0, editableRect.width, editableRect.height, editableRect._m)) { // right edge
                    editableRect.setCursorShape(Qt.SizeHorCursor)
                } else if (editableRect.isPointNearLine(pt.x, pt.y, 0, 0, editableRect.width, 0, editableRect._m)) { // top edge
                    editableRect.setCursorShape(Qt.SizeVerCursor)
                } else if (editableRect.isPointNearLine(pt.x, pt.y, 0, editableRect.height, editableRect.width, editableRect.height, editableRect._m)) { // top edge
                    editableRect.setCursorShape(Qt.SizeVerCursor)
                } else {
                    editableRect.setCursorShape(Qt.SizeAllCursor)
                }
            }
            dragType = -1
        }

        onPositionChanged: function(mouse) {
            var pt = mapToItem(editableRect, mouse.x, mouse.y)
            var pos = mapToItem(_image, mouse.x, mouse.y)
            if (dragEnable) {

            } else if (editableRect.selected) {
                if (dragType === 0) { // top left
                    editableRect.updateByTopLeft(pos)
                } else if (dragType === 1) { // bottom right
                    editableRect.updateByBottomRight(pos)
                } else if (dragType === 2) { // bottom left
                    editableRect.updateByBottomLeft(pos)
                } else if (dragType === 3) { // top right
                    editableRect.updateByTopRight(pos)
                } else if (dragType === 4) { // left edge
                    editableRect.updateByLeft(pos)
                } else if (dragType === 5) { // right edge
                    editableRect.updateByRight(pos)
                } else if (dragType === 6) { // top edge
                    editableRect.updateByTop(pos)
                } else if (dragType === 7) { // bottom edge
                    editableRect.updateByBottom(pos)
                } else if (editableRect.isPointNearPoint(pt.x, pt.y, 0, 0, editableRect._m)) { // top left
                    editableRect.setCursorShape(Qt.SizeFDiagCursor)
                    if (mouse.buttons & Qt.LeftButton) {
                        dragType = 0
                        editableRect.updateByTopLeft(pos)
                    }
                } else if (editableRect.isPointNearPoint(pt.x, pt.y, editableRect.width, editableRect.height, editableRect._m)) { // bottom right
                    editableRect.setCursorShape(Qt.SizeFDiagCursor)
                    if (mouse.buttons & Qt.LeftButton) {
                        dragType = 1
                        editableRect.updateByBottomRight(pos)
                    }
                } else if (editableRect.isPointNearPoint(pt.x, pt.y, 0, editableRect.height, editableRect._m)) { // bottom left
                    editableRect.setCursorShape(Qt.SizeBDiagCursor)
                    if (mouse.buttons & Qt.LeftButton) {
                        dragType = 2
                        editableRect.updateByBottomLeft(pos)
                    }
                } else if (editableRect.isPointNearPoint(pt.x, pt.y, editableRect.width, 0, editableRect._m)) { // top right
                    editableRect.setCursorShape(Qt.SizeBDiagCursor)
                    if (mouse.buttons & Qt.LeftButton) {
                        dragType = 3
                        editableRect.updateByTopRight(pos)
                    }
                } else if (editableRect.isPointNearLine(pt.x, pt.y, 0, 0, 0, editableRect.height, editableRect._m) && pt.y > 0 && pt.y < editableRect.height) { // left edge
                    editableRect.setCursorShape(Qt.SizeHorCursor)
                    if (mouse.buttons & Qt.LeftButton) {
                        dragType = 4
                        editableRect.updateByLeft(pos)
                    }
                } else if (editableRect.isPointNearLine(pt.x, pt.y, editableRect.width, 0, editableRect.width, editableRect.height, editableRect._m) && pt.y > 0 && pt.y < editableRect.height) { // right edge
                    editableRect.setCursorShape(Qt.SizeHorCursor)
                    if (mouse.buttons & Qt.LeftButton) {
                        dragType = 5
                        editableRect.updateByRight(pos)
                    }
                } else if (editableRect.isPointNearLine(pt.x, pt.y, 0, 0, editableRect.width, 0, editableRect._m) && pt.x > 0 && pt.x < editableRect.width) { // top edge
                    editableRect.setCursorShape(Qt.SizeVerCursor)
                    if (mouse.buttons & Qt.LeftButton) {
                        dragType = 6
                        editableRect.updateByTop(pos)
                    }
                } else if (editableRect.isPointNearLine(pt.x, pt.y, 0, editableRect.height, editableRect.width, editableRect.height, editableRect._m) && pt.x > 0 && pt.x < editableRect.width) { // bottom edge
                    editableRect.setCursorShape(Qt.SizeVerCursor)
                    if (mouse.buttons & Qt.LeftButton) {
                        dragType = 7
                        editableRect.updateByBottom(pos)
                    }
                } else {
                    editableRect.setCursorShape(Qt.SizeAllCursor)
                }
            } else {
                editableRect.setCursorShape(Qt.ArrowCursor)
            }
        }
    }

    function setCursorShape(cursorShape) {
        if (mouseArea.cursorShape !== cursorShape) {
            mouseArea.cursorShape = cursorShape
        }
    }

    function isPointNearPoint(px, py, cx, cy, radius) {
        var dx = px - cx
        var dy = py - cy
        var dist = Math.sqrt(dx * dx + dy * dy)
        return dist < radius
    }

    function isPointNearLine(px, py, x1, y1, x2, y2, dist) {
        var distance = Math.abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1) / Math.sqrt((y2 - y1) * (y2 - y1) + (x2 - x1) * (x2 - x1))
        return distance < dist
    }

    function updateByTopLeft(pos) {
        var top = pos.y
        var left = pos.x
        var right = editableRect.x + editableRect.width
        var bottom = editableRect.y + editableRect.height
        top = Math.max(0, Math.min(top, bottom - editableRect.minimalSize))
        left = Math.max(0, Math.min(left, right - editableRect.minimalSize))
        editableRect.x = left
        editableRect.y = top
        editableRect.width = right - left
        editableRect.height = bottom - top
    }

    function updateByTopRight(pos) {
        var top = pos.y
        var left = editableRect.x
        var right = pos.x
        var bottom = editableRect.y + editableRect.height
        top = Math.max(0, Math.min(top, bottom - editableRect.minimalSize))
        right = Math.min(Math.max(right, left + editableRect.minimalSize), editableRect.parent.width)
        editableRect.x = left
        editableRect.y = top
        editableRect.width = right - left
        editableRect.height = bottom - top
    }

    function updateByBottomLeft(pos) {
        var top = editableRect.y
        var left = pos.x
        var right = editableRect.x + editableRect.width
        var bottom = pos.y
        bottom = Math.min(Math.max(bottom, top + editableRect.minimalSize), editableRect.parent.height)
        left = Math.max(0, Math.min(left, right - editableRect.minimalSize))
        editableRect.x = left
        editableRect.y = top
        editableRect.width = right - left
        editableRect.height = bottom - top
    }

    function updateByBottomRight(pos) {
        var top = editableRect.y
        var left = editableRect.x
        var right = pos.x
        var bottom = pos.y
        bottom = Math.min(Math.max(bottom, top + editableRect.minimalSize), editableRect.parent.height)
        right = Math.min(Math.max(right, left + editableRect.minimalSize), editableRect.parent.width)
        editableRect.x = left
        editableRect.y = top
        editableRect.width = right - left
        editableRect.height = bottom - top
    }

    function updateByTop(pos) {
        var top = pos.y
        var bottom = editableRect.y + editableRect.height
        top = Math.max(0, Math.min(top, bottom - editableRect.minimalSize))
        editableRect.y = top
        editableRect.height = bottom - top
    }

    function updateByBottom(pos) {
        var top = editableRect.y
        var bottom = pos.y
        bottom = Math.min(Math.max(bottom, top + editableRect.minimalSize), editableRect.parent.height)
        editableRect.y = top
        editableRect.height = bottom - top
    }

    function updateByLeft(pos) {
        var left = pos.x
        var right = editableRect.x + editableRect.width
        left = Math.max(0, Math.min(left, right - editableRect.minimalSize))
        editableRect.x = left
        editableRect.width = right - left
    }

    function updateByRight(pos) {
        var left = editableRect.x
        var right = pos.x
        right = Math.min(Math.max(right, left + editableRect.minimalSize), editableRect.parent.width)
        editableRect.x = left
        editableRect.width = right - left
    }

    function updateByData(data) {
        if (data.length < 4)
            return
        editableRect.x = data[0]
        editableRect.y = data[1]
        editableRect.width = data[2]
        editableRect.height = data[3]
    }

    function clear() {
        editableRect.x = 0
        editableRect.y = 0
        editableRect.width = 0
        editableRect.height = 0
        editableRect.visible = false
    }
}

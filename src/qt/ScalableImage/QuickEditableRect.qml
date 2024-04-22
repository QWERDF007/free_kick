import QtQuick
import QtQuick.Controls

import ScalableImage

QuickRectangle {
    id: editableRect
    property bool selected: false
    property real _m: Math.max(5, 10 / parent.scale) // 靠近顶点和边的距离阈值
    property int _ms: 5 // 矩形最小大小
    property var roiData: [] // 矩形数据
    property point startPoint // 绘制七点, 坐标系是 parent 上的


    color: "transparent"
    border.color: "red"
    border.width: selected ? Math.max(2, 2 / parent.scale) : Math.max(1, 1 / parent.scale)

    property QtObject inner: QtObject {
        property real left: 0
        property real right: editableRect.width
        property real top: 0
        property real bottom: editableRect.height
    }

    onXChanged: {
        if (mouseArea.dragEnable) {
            roiData = [editableRect.x, editableRect.y, editableRect.width, editableRect.height]
        }
    }

    onYChanged: {
        if (mouseArea.dragEnable) {
            roiData = [editableRect.x, editableRect.y, editableRect.width, editableRect.height]
        }
    }



    MouseArea {
        id: mouseArea
        property bool dragEnable: false
        property int dragType: -1
        anchors.fill: parent
        anchors.margins: editableRect.visible ? -10 : 0 // 鼠标区域比矩形大好处理顶点和边的编辑
        acceptedButtons: Qt.AllButtons
        hoverEnabled: true
        drag.target: dragEnable ? editableRect : null
        drag.minimumX: 0
        drag.maximumX: editableRect.parent.width - editableRect.width
        drag.minimumY: 0
        drag.maximumY: editableRect.parent.height - editableRect.height
        onPressed: function(mouse) {
            if (mouse.button === Qt.LeftButton) {
                var pt = mapToItem(editableRect, mouse.x, mouse.y)
                if (editableRect.selected && pt.x > inner.left + editableRect._m && pt.x < inner.right - editableRect._m && pt.y > inner.top + editableRect._m && pt.y < inner.bottom - editableRect._m) {
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
                if (editableRect.isPointNearPoint(pt.x, pt.y, inner.left, inner.top, editableRect._m)) { // top left
                    editableRect.setCursorShape(Qt.SizeFDiagCursor)
                } else if (editableRect.isPointNearPoint(pt.x, pt.y, inner.right, inner.bottom, editableRect._m)) { // bottom right
                    editableRect.setCursorShape(Qt.SizeFDiagCursor)
                } else if (editableRect.isPointNearPoint(pt.x, pt.y, inner.left, inner.bottom, editableRect._m)) { // bottom left
                    editableRect.setCursorShape(Qt.SizeBDiagCursor)
                } else if (editableRect.isPointNearPoint(pt.x, pt.y, inner.right, inner.top, editableRect._m)) { // top right
                    editableRect.setCursorShape(Qt.SizeBDiagCursor)
                } else if (editableRect.isPointNearLine(pt.x, pt.y, inner.left, inner.top, inner.left, inner.bottom, editableRect._m)) { // left edge
                    editableRect.setCursorShape(Qt.SizeHorCursor)
                } else if (editableRect.isPointNearLine(pt.x, pt.y, inner.right, inner.top, inner.right, inner.bottom, editableRect._m)) { // right edge
                    editableRect.setCursorShape(Qt.SizeHorCursor)
                } else if (editableRect.isPointNearLine(pt.x, pt.y, inner.left, inner.top, inner.right, inner.top, editableRect._m)) { // top edge
                    editableRect.setCursorShape(Qt.SizeVerCursor)
                } else if (editableRect.isPointNearLine(pt.x, pt.y, inner.left, inner.bottom, inner.right, inner.bottom, editableRect._m)) { // bottom edge
                    editableRect.setCursorShape(Qt.SizeVerCursor)
                } else {
                    if (pt.x > inner.left && pt.x < inner.right && pt.y > inner.top && pt.y < inner.bottom) {
                        editableRect.setCursorShape(Qt.SizeAllCursor)
                    } else {
                        editableRect.setCursorShape(Qt.ArrowCursor)
                    }
                }
            }
            dragType = -1
        }

        onPositionChanged: function(mouse) {
            var pt = mapToItem(editableRect, mouse.x, mouse.y) // 用于判断是否处理 editableRect 的顶点和边
            var pos = mapToItem(editableRect.parent, mouse.x, mouse.y) // 用于更新 editableRect
            // console.log("dist", editableRect.distanceToLine(pt.x, pt.y, editableRect.width, 0, editableRect.width, editableRect.height))
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
                } else if (editableRect.isPointNearPoint(pt.x, pt.y, inner.left, inner.top, editableRect._m)) { // top left
                    editableRect.setCursorShape(Qt.SizeFDiagCursor)
                    if (mouse.buttons & Qt.LeftButton) {
                        dragType = 0
                        editableRect.updateByTopLeft(pos)
                    }
                } else if (editableRect.isPointNearPoint(pt.x, pt.y, inner.right, inner.bottom, editableRect._m)) { // bottom right
                    editableRect.setCursorShape(Qt.SizeFDiagCursor)
                    if (mouse.buttons & Qt.LeftButton) {
                        dragType = 1
                        editableRect.updateByBottomRight(pos)
                    }
                } else if (editableRect.isPointNearPoint(pt.x, pt.y, inner.left, inner.bottom, editableRect._m)) { // bottom left
                    editableRect.setCursorShape(Qt.SizeBDiagCursor)
                    if (mouse.buttons & Qt.LeftButton) {
                        dragType = 2
                        editableRect.updateByBottomLeft(pos)
                    }
                } else if (editableRect.isPointNearPoint(pt.x, pt.y, inner.right, inner.top, editableRect._m)) { // top right
                    editableRect.setCursorShape(Qt.SizeBDiagCursor)
                    if (mouse.buttons & Qt.LeftButton) {
                        dragType = 3
                        editableRect.updateByTopRight(pos)
                    }
                } else if (editableRect.isPointNearLine(pt.x, pt.y, inner.left, inner.top, inner.left, inner.bottom, editableRect._m) && pt.y > 0 && pt.y < editableRect.height) { // left edge
                    editableRect.setCursorShape(Qt.SizeHorCursor)
                    if (mouse.buttons & Qt.LeftButton) {
                        dragType = 4
                        editableRect.updateByLeft(pos)
                    }
                } else if (editableRect.isPointNearLine(pt.x, pt.y, inner.right, inner.top, inner.right, inner.bottom, editableRect._m) && pt.y > 0 && pt.y < editableRect.height) { // right edge
                    editableRect.setCursorShape(Qt.SizeHorCursor)
                    if (mouse.buttons & Qt.LeftButton) {
                        dragType = 5
                        editableRect.updateByRight(pos)
                    }
                } else if (editableRect.isPointNearLine(pt.x, pt.y, inner.left, inner.top, inner.right, inner.top, editableRect._m) && pt.x > 0 && pt.x < editableRect.width) { // top edge
                    editableRect.setCursorShape(Qt.SizeVerCursor)
                    if (mouse.buttons & Qt.LeftButton) {
                        dragType = 6
                        editableRect.updateByTop(pos)
                    }
                } else if (editableRect.isPointNearLine(pt.x, pt.y, inner.left, inner.bottom, inner.right, inner.bottom, editableRect._m) && pt.x > 0 && pt.x < editableRect.width) { // bottom edge
                    editableRect.setCursorShape(Qt.SizeVerCursor)
                    if (mouse.buttons & Qt.LeftButton) {
                        dragType = 7
                        editableRect.updateByBottom(pos)
                    }
                } else {
                    if (pt.x > inner.left && pt.x < inner.right && pt.y > inner.top && pt.y < inner.bottom) {
                        editableRect.setCursorShape(Qt.SizeAllCursor)
                    } else {
                        editableRect.setCursorShape(Qt.ArrowCursor)
                    }
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

    function distanceToPoint(px, py, cx, cy) {
        var dx = px - cx
        var dy = py - cy
        return Math.sqrt(dx * dx + dy * dy)
    }

    function isPointNearPoint(px, py, cx, cy, radius) {
        var distance = distanceToPoint(px, py, cx, cy)
        return distance < radius
    }

    function distanceToLine(px, py, x1, y1, x2, y2) {
        return Math.abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1) / (Math.sqrt((y2 - y1) * (y2 - y1) + (x2 - x1) * (x2 - x1)) + 1e-9)
    }

    function isPointNearLine(px, py, x1, y1, x2, y2, dist) {
        var distance = distanceToLine(px, py, x1, y1, x2, y2)
        return distance < dist
    }

    /**
     * @brief 更新数据
     * @param data 矩形数据, xywh 格式
     */
    function updateByData(data) {
        if (data.length < 4)
            return
        editableRect.x = data[0]
        editableRect.y = data[1]
        editableRect.width = data[2]
        editableRect.height = data[3]
        editableRect.roiData = data
        if (!editableRect.visible)
            editableRect.visible = true
    }

    /**
     * @brief 拖拽矩形的左上顶点更新
     * @param pos parent 坐标系的鼠标位置
     */
    function updateByTopLeft(pos) {
        var top = pos.y
        var left = pos.x
        var right = editableRect.x + editableRect.width
        var bottom = editableRect.y + editableRect.height
        top = Math.max(0, Math.min(top, bottom - editableRect._ms))
        left = Math.max(0, Math.min(left, right - editableRect._ms))
        updateByData([left, top, right - left, bottom - top])
    }

    /**
     * @brief 拖拽矩形的右上顶点更新
     * @param pos parent 坐标系的鼠标位置
     */
    function updateByTopRight(pos) {
        var top = pos.y
        var left = editableRect.x
        var right = pos.x
        var bottom = editableRect.y + editableRect.height
        top = Math.max(0, Math.min(top, bottom - editableRect._ms))
        right = Math.min(Math.max(right, left + editableRect._ms), editableRect.parent.width)
        updateByData([editableRect.x, top, right - left , bottom - top])
    }

    /**
     * @brief 拖拽矩形的左下顶点更新
     * @param pos parent 坐标系的鼠标位置
     */
    function updateByBottomLeft(pos) {
        var top = editableRect.y
        var left = pos.x
        var right = editableRect.x + editableRect.width
        var bottom = pos.y
        bottom = Math.min(Math.max(bottom, top + editableRect._ms), editableRect.parent.height)
        left = Math.max(0, Math.min(left, right - editableRect._ms))
        updateByData([left, editableRect.y, right - left, bottom - top])
    }

    /**
     * @brief 拖拽矩形的右下顶点更新
     * @param pos parent 坐标系的鼠标位置
     */
    function updateByBottomRight(pos) {
        var top = editableRect.y
        var left = editableRect.x
        var right = pos.x
        var bottom = pos.y
        bottom = Math.min(Math.max(bottom, top + editableRect._ms), editableRect.parent.height)
        right = Math.min(Math.max(right, left + editableRect._ms), editableRect.parent.width)
        updateByData([editableRect.x, editableRect.y, right - left, bottom - top])
    }

    /**
     * @brief 拖拽矩形的上边更新
     * @param pos parent 坐标系的鼠标位置
     */
    function updateByTop(pos) {
        var top = pos.y
        var bottom = editableRect.y + editableRect.height
        top = Math.max(0, Math.min(top, bottom - editableRect._ms))
        updateByData([editableRect.x, top, editableRect.width, bottom - top])
    }

    /**
     * @brief 拖拽矩形的下边更新
     * @param pos parent 坐标系的鼠标位置
     */
    function updateByBottom(pos) {
        var top = editableRect.y
        var bottom = pos.y
        bottom = Math.min(Math.max(bottom, top + editableRect._ms), editableRect.parent.height)
        updateByData([editableRect.x, editableRect.y, editableRect.width, bottom - top])
    }

    /**
     * @brief 拖拽矩形的左边更新
     * @param pos parent 坐标系的鼠标位置
     */
    function updateByLeft(pos) {
        var left = pos.x
        var right = editableRect.x + editableRect.width
        left = Math.max(0, Math.min(left, right - editableRect._ms))
        updateByData([left, editableRect.y, right - left, editableRect.height])
    }

    /**
     * @brief 拖拽矩形的右边更新
     * @param pos parent 坐标系的鼠标位置
     */
    function updateByRight(pos) {
        var left = editableRect.x
        var right = pos.x
        right = Math.min(Math.max(right, left + editableRect._ms), editableRect.parent.width)
        updateByData([editableRect.x, editableRect.y, right - left, editableRect.height])
    }



    /**
     * @brief 重置矩形, 清除数据
     */
    function clear() {
        editableRect.x = 0
        editableRect.y = 0
        editableRect.width = 0
        editableRect.height = 0
        editableRect.visible = false
        editableRect.selected = false
        editableRect.roiData = []
    }

    /**
     * @brief 在 pos 处更新矩形
     * @param pos 鼠标在 parent 坐标系的位置
     */
    function updateByPos(pos) {
        var pt1 = editableRect.startPoint
        var pt2 = pos
        var left = Math.min(pt1.x, pt2.x)
        left = Math.max(0, left)
        var right = Math.max(pt1.x, pt2.x)
        right = Math.min(right, editableRect.parent.width)
        var top = Math.min(pt1.y, pt2.y)
        top = Math.max(0, top)
        var bottom = Math.max(pt1.y, pt2.y)
        bottom = Math.min(bottom, editableRect.parent.height)
        var x = left
        var y = top
        var width = right - left
        var height = bottom - top
        if (width > 0 && height > 0) {
            editableRect.updateByData([x, y, width, height])
        }
    }
}

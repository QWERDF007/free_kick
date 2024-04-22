import QtQuick
import QtQuick.Controls

import ScalableImage

QuickCircle {
    id: editableCircle
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
        property real right: editableCircle.width
        property real top: 0
        property real bottom: editableCircle.height
    }

    onCenterChanged: {
        if (visible) {
            roiData = [editableCircle.center.x, editableCircle.center.y, editableCircle.radius]
        }
    }



    MouseArea {
        id: mouseArea
        property bool dragEnable: false
        property int dragType: -1
        anchors.fill: parent
        anchors.margins: editableCircle.visible ? -5 : 0 // 鼠标区域比矩形大好处理顶点和边的编辑
        acceptedButtons: Qt.AllButtons
        hoverEnabled: true
        drag.target: dragEnable ? editableCircle : null
        drag.minimumX: 0
        drag.maximumX: editableCircle.parent.width - editableCircle.width
        drag.minimumY: 0
        drag.maximumY: editableCircle.parent.height - editableCircle.height
        onPressed: function(mouse) {
            if (mouse.button === Qt.LeftButton) {
                var pos = mapToItem(editableCircle.parent, mouse.x, mouse.y)
                var d = distanceToPoint(pos.x, pos.y, editableCircle.center.x, editableCircle.center.y)
                if (d < editableCircle.radius - editableCircle._m) {
                    dragEnable = true
                }
            } else if (mouse.button === Qt.MiddleButton) {
                mouse.accepted = false
            }
        }

        onReleased: function(mouse) {
            if (mouse.button === Qt.LeftButton) {
                dragEnable = false
                editableCircle.selected = true
                var pos = mapToItem(editableCircle.parent, mouse.x, mouse.y) // 用于更新 editableCircle
                if (isPointNearEdge(pos.x, pos.y, editableCircle.center.x, editableCircle.center.y, editableCircle._m)) {
                    var dx = pos.x - editableCircle.center.x
                    var dy = pos.y - editableCircle.center.y
                    var adx = Math.abs(dx)
                    var ady = Math.abs(dy)
                    if (adx > ady) {
                        editableCircle.setCursorShape(Qt.SizeHorCursor)
                    } else {
                        editableCircle.setCursorShape(Qt.SizeVerCursor)
                    }
                } else {
                    editableCircle.setCursorShape(Qt.SizeAllCursor)
                }
            }
            dragType = -1
        }

        onPositionChanged: function(mouse) {
            var pos = mapToItem(editableCircle.parent, mouse.x, mouse.y) // 用于更新 editableCircle
            if (dragEnable) {

            } else if (editableCircle.selected) {
                if (dragType === 0) { // left or right
                    editableCircle.updateByLeftRight(pos)
                }
                else if (dragType === 1) { // top or bottom
                    editableCircle.updateByTopBottom(pos)
                }
                else if (isPointNearEdge(pos.x, pos.y, editableCircle.center.x, editableCircle.center.y, editableCircle._m)) {
                    var dx = pos.x - editableCircle.center.x
                    var dy = pos.y - editableCircle.center.y
                    var adx = Math.abs(dx)
                    var ady = Math.abs(dy)
                    if (adx > ady) {
                        editableCircle.setCursorShape(Qt.SizeHorCursor)
                        if (mouse.buttons & Qt.LeftButton) {
                            dragType = 0
                            editableCircle.updateByLeftRight(pos)
                        }
                    } else {
                        editableCircle.setCursorShape(Qt.SizeVerCursor)
                        if (mouse.buttons & Qt.LeftButton) {
                            dragType = 1
                            editableCircle.updateByTopBottom(pos)
                        }
                    }
                } else {
                    editableCircle.setCursorShape(Qt.SizeAllCursor)
                }
            } else {
                editableCircle.setCursorShape(Qt.ArrowCursor)
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

    function isPointNearEdge(px, py, cx, cy, threshold) {
        var distance = distanceToPoint(px, py, cx, cy)
        return Math.abs(distance - editableCircle.radius) < threshold
    }


    /**
     * @brief 更新数据
     * @param data 矩形数据, xyr 格式
     */
    function updateByData(data) {
        if (data.length < 3)
            return
        editableCircle.center = Qt.point(data[0], data[1])
        editableCircle.radius = data[2]
        if (!editableCircle.visible) {
            editableCircle.visible = true
        }
    }

    /**
     * @brief 拖拽矩形的左/右边更新
     * @param pos parent 坐标系的鼠标位置
     */
    function updateByLeftRight(pos) {
        var radius = Math.abs(pos.x - editableCircle.center.x)
        if (radius > editableCircle.center.x) {
            radius = editableCircle.center.x
        }
        if (editableCircle.center.x + radius > editableCircle.parent.width) {
            radius = editableCircle.parent.width - editableCircle.center.x
        }
        if (radius > editableCircle.center.y) {
            radius = editableCircle.center.y
        }
        if (editableCircle.center.y + radius > editableCircle.parent.height) {
            radius = editableCircle.parent.height - editableCircle.center.y
        }
        updateByData([editableCircle.center.x, editableCircle.center.y, Math.max(radius, editableCircle._m)])
    }

    /**
     * @brief 拖拽矩形的上/下边更新
     * @param pos parent 坐标系的鼠标位置
     */
    function updateByTopBottom(pos) {
        var radius = Math.abs(pos.y - editableCircle.center.y)
        if (radius > editableCircle.center.x) {
            radius = editableCircle.center.x
        }
        if (editableCircle.center.x + radius > editableCircle.parent.width) {
            radius = editableCircle.parent.width - editableCircle.center.x
        }
        if (radius > editableCircle.center.y) {
            radius = editableCircle.center.y
        }
        if (editableCircle.center.y + radius > editableCircle.parent.height) {
            radius = editableCircle.parent.height - editableCircle.center.y
        }
        updateByData([editableCircle.center.x, editableCircle.center.y, Math.max(radius, editableCircle._m)])
    }

    /**
     * @brief 重置矩形, 清除数据
     */
    function clear() {
        editableCircle.center = Qt.point(0, 0)
        editableCircle.radius = 0
        editableCircle.visible = false
        editableCircle.selected = false
        editableCircle.roiData = []
    }

    /**
     * @brief 在 pos 处更新圆
     * @param pos 鼠标在 parent 坐标系的位置
     */
    function updateByPos(pos) {
        var pt1 = editableCircle.startPoint
        var pt2 = pos
        var dx = Math.abs(pt2.x - pt1.x)
        var dy = Math.abs(pt2.y - pt1.y)
        var x = pt1.x
        var y = pt1.y
        var radius = Math.max(dx, dy)
        if (radius > x)
            radius = x
        if (radius > y)
            radius = y
        if (x + radius > editableCircle.parent.width)
            radius = editableCircle.parent.width - x
        if (y + radius > editableCircle.parent.height)
            radius = editableCircle.parent.height - y
        if (radius > editableCircle._m) {
            editableCircle.updateByData([x, y, radius])
        }
    }
}

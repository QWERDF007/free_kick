import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Qt5Compat.GraphicalEffects
import QtQuick.Shapes

Item {
    id: scalableImage
    clip: true
    width: 200
    height: 200
    // implicitHeight: _image.implicitHeight // 不确定绑定Image的隐式宽高有没有问题
    // implicitWidth: _image.implicitWidth

    property alias image: _image
    property alias status: _image.status
    property alias source: _image.source
    property alias sourceSize: _image.sourceSize
    property bool imageDragEnable: false
    property real stepSize: {
        if (_image.scale < 2) {
            return 0.1
        } else if (_image.scale < 10) {
            return 1
        } else {
            return 2
        }
    }

    property real from: 0.1
    property real to: 32
    property var scaledImagePos: mapFromItem(_image, 0, 0)
    property bool isFitInView: true
    property real imageSourceScale: {
        if (_image.source !== "" && _image.status === _image.Ready) {
            return Math.min(scalableImage.height / _image.sourceSize.height, scalableImage.width / _image.sourceSize.width)
        }
        return 1.0
    }

    property point startPoint
    property color drawingColor: "red"
    property bool drawing: false

    Component.onCompleted: {
        _image.source = "file:///H:/Datasets/性感美女/raw/1691766509-cc387fc3be7ab7d.jpg"
    }

    MouseArea {
        id: mouseArea
        anchors.fill: parent
        drag.target: scalableImage.imageDragEnable ? _image : null
        drag.axis: Drag.XAndYAxis
        hoverEnabled: true
        acceptedButtons: Qt.AllButtons

        onPressed: function (mouse) {
            scalableImage.forceActiveFocus()
            if (mouse.button === Qt.LeftButton) {
                if (mouse.modifiers & Qt.ControlModifier) {
                    setImageDragEnable(true)
                    setCursorShape(Qt.ClosedHandCursor)
                } else {
                    roi_rect.visible = true
                    roi_rect.selected = false
                    scalableImage.startDrawingRect(mouse)
                }
            } else if (mouse.button === Qt.MiddleButton) {
                setImageDragEnable(true)
                setCursorShape(Qt.ClosedHandCursor)
                roi_rect.setCursorShape(Qt.ClosedHandCursor)
            }
        }

        onReleased: function (mouse) {
            if (scalableImage.imageDragEnable) {
                setImageDragEnable(false)
                if (mouse.modifiers & Qt.ControlModifier) {
                    setCursorShape(Qt.OpenHandCursor)
                } else {
                    setCursorShape(Qt.ArrowCursor)
                    if (roi_rect.selected) {
                        roi_rect.setCursorShape(Qt.SizeAllCursor)
                    } else {
                        roi_rect.setCursorShape(Qt.ArrowCursor)
                    }
                }
            } else if (mouse.button === Qt.LeftButton) {
                scalableImage.updateRect(mouse)
                scalableImage.drawing = false
            }
        }

        onPositionChanged: function (mouse) {
            if (scalableImage.drawing) {
                scalableImage.updateRect(mouse)
                scalableImage.updateDrawingRectByMouse(mouse)
            }
        }

        onWheel: function (wheel) {
            scaleImageByWheel(wheel)
            updateImagePos()
        }
    }

    Keys.onPressed: function(event) {
        if (event.key === Qt.Key_Control) {
            if (mouseArea.containsPress) {
                setCursorShape(Qt.ClosedHandCursor)
            } else {
                setCursorShape(Qt.OpenHandCursor)
            }
        } else if (event.key === Qt.Key_Space) {
            fitInView()
        } else if (event.key === Qt.Key_Escape) {
            scalableImage.clearROI()
        }
    }

    Keys.onReleased: function(event) {
        if (event.key === Qt.Key_Control && !mouseArea.containsPress) {
            setCursorShape(Qt.ArrowCursor)
        }
    }

    Image {
        id: _image
        smooth: false
        asynchronous: true // 异步加载会导致自适应窗口出问题
        property real xOffset: Math.abs(width - paintedWidth) / 2 * scale
        property real yOffset: Math.abs(height - paintedHeight) / 2 * scale
        fillMode: Image.PreserveAspectFit
        onXChanged: {
            updateImagePos()
        }
        onYChanged: {
            updateImagePos()
        }
        onStatusChanged: {
            if (_image.status === Image.Ready) {
                fitInView()
            }
        }
        transformOrigin: Item.TopLeft

        onScaleChanged: {
            console.log("scale", scale)
        }

        Rectangle {
            id: roi_rect
            property bool selected: false
            property real _opacity: 0.3
            property real _m: 10 / parent.scale
            property int minimalSize: 5
            opacity: selected ? _opacity * 2 : _opacity
            // visible: false
            border.color: "red"
            border.width: selected ? 2 : 1
            color: "transparent"
            // color: "red"
            MouseArea {
                id: roi_rect_mouse
                property bool dragEnable: false
                property int dragType: -1
                anchors.fill: parent
                anchors.margins: -10
                acceptedButtons: Qt.AllButtons
                hoverEnabled: true
                drag.target: dragEnable ? roi_rect : null
                drag.minimumX: 0
                drag.maximumX: _image.width - roi_rect.width
                drag.minimumY: 0
                drag.maximumY: _image.height - roi_rect.height
                onPressed: function(mouse) {
                    if (mouse.button === Qt.LeftButton) {
                        roi_rect.selected = true
                        var pt = mapToItem(roi_rect, mouse.x, mouse.y)
                        if (roi_rect.isPointNearPoint(pt.x, pt.y, 0, 0, roi_rect._m)) { // top left
                            roi_rect.setCursorShape(Qt.SizeFDiagCursor)
                        } else if (roi_rect.isPointNearPoint(pt.x, pt.y, roi_rect.width, roi_rect.height, roi_rect._m)) { // bottom right
                            roi_rect.setCursorShape(Qt.SizeFDiagCursor)
                        } else if (roi_rect.isPointNearPoint(pt.x, pt.y, 0, roi_rect.height, roi_rect._m)) { // bottom left
                            roi_rect.setCursorShape(Qt.SizeBDiagCursor)
                        } else if (roi_rect.isPointNearPoint(pt.x, pt.y, roi_rect.width, 0, roi_rect._m)) { // top right
                            roi_rect.setCursorShape(Qt.SizeBDiagCursor)
                        } else if (roi_rect.isPointNearLine(pt.x, pt.y, 0, 0, 0, roi_rect.height, roi_rect._m)) { // left edge
                            roi_rect.setCursorShape(Qt.SizeHorCursor)
                        } else if (roi_rect.isPointNearLine(pt.x, pt.y, roi_rect.width, 0, roi_rect.width, roi_rect.height, roi_rect._m)) { // right edge
                            roi_rect.setCursorShape(Qt.SizeHorCursor)
                        } else if (roi_rect.isPointNearLine(pt.x, pt.y, 0, 0, roi_rect.width, 0, roi_rect._m)) { // top edge
                            roi_rect.setCursorShape(Qt.SizeVerCursor)
                        } else if (roi_rect.isPointNearLine(pt.x, pt.y, 0, roi_rect.height, roi_rect.width, roi_rect.height, roi_rect._m)) { // top edge
                            roi_rect.setCursorShape(Qt.SizeVerCursor)
                        } else {
                            roi_rect.setCursorShape(Qt.SizeAllCursor)
                            dragEnable = true
                        }
                    } else if (mouse.button === Qt.MiddleButton) {
                        mouse.accepted = false
                    }
                }

                onReleased: function(mouse) {
                    if (mouse.button === Qt.LeftButton) {
                        dragEnable = false
                    }
                    dragType = -1
                }



                onPositionChanged: function(mouse) {
                    // console.log("onPositionChanged", mouse.x, mouse.y, dragType)
                    var pt = mapToItem(roi_rect, mouse.x, mouse.y)
                    var pos = mapToItem(_image, mouse.x, mouse.y)
                    if (dragEnable) {

                    } else if (roi_rect.selected) {
                        if (dragType === 0) { // top left
                            roi_rect.updateByTopLeft(pos)
                        } else if (dragType === 1) { // bottom right
                            roi_rect.updateByBottomRight(pos)
                        } else if (dragType === 2) { // bottom left
                            roi_rect.updateByBottomLeft(pos)
                        } else if (dragType === 3) { // top right
                            roi_rect.updateByTopRight(pos)
                        } else if (dragType === 4) { // left edge
                            roi_rect.updateByLeft(pos)
                        } else if (dragType === 5) { // right edge
                            roi_rect.updateByRight(pos)
                        } else if (dragType === 6) { // top edge
                            roi_rect.updateByTop(pos)
                        } else if (dragType === 7) { // bottom edge
                            roi_rect.updateByBottom(pos)
                        } else if (roi_rect.isPointNearPoint(pt.x, pt.y, 0, 0, roi_rect._m)) { // top left
                            roi_rect.setCursorShape(Qt.SizeFDiagCursor)
                            if (mouse.buttons & Qt.LeftButton) {
                                dragType = 0
                                roi_rect.updateByTopLeft(pos)
                            }
                        } else if (roi_rect.isPointNearPoint(pt.x, pt.y, roi_rect.width, roi_rect.height, roi_rect._m)) { // bottom right
                            roi_rect.setCursorShape(Qt.SizeFDiagCursor)
                            if (mouse.buttons & Qt.LeftButton) {
                                dragType = 1
                                roi_rect.updateByBottomRight(pos)
                            }
                        } else if (roi_rect.isPointNearPoint(pt.x, pt.y, 0, roi_rect.height, roi_rect._m)) { // bottom left
                            roi_rect.setCursorShape(Qt.SizeBDiagCursor)
                            if (mouse.buttons & Qt.LeftButton) {
                                dragType = 2
                                roi_rect.updateByBottomLeft(pos)
                            }
                        } else if (roi_rect.isPointNearPoint(pt.x, pt.y, roi_rect.width, 0, roi_rect._m)) { // top right
                            roi_rect.setCursorShape(Qt.SizeBDiagCursor)
                            if (mouse.buttons & Qt.LeftButton) {
                                dragType = 3
                                roi_rect.updateByTopRight(pos)
                            }
                        } else if (roi_rect.isPointNearLine(pt.x, pt.y, 0, 0, 0, roi_rect.height, roi_rect._m) && pt.y > 0 && pt.y < roi_rect.height) { // left edge
                            roi_rect.setCursorShape(Qt.SizeHorCursor)
                            if (mouse.buttons & Qt.LeftButton) {
                                dragType = 4
                                roi_rect.updateByLeft(pos)
                            }
                        } else if (roi_rect.isPointNearLine(pt.x, pt.y, roi_rect.width, 0, roi_rect.width, roi_rect.height, roi_rect._m) && pt.y > 0 && pt.y < roi_rect.height) { // right edge
                            roi_rect.setCursorShape(Qt.SizeHorCursor)
                            if (mouse.buttons & Qt.LeftButton) {
                                dragType = 5
                                roi_rect.updateByRight(pos)
                            }
                        } else if (roi_rect.isPointNearLine(pt.x, pt.y, 0, 0, roi_rect.width, 0, roi_rect._m) && pt.x > 0 && pt.x < roi_rect.width) { // top edge
                            roi_rect.setCursorShape(Qt.SizeVerCursor)
                            if (mouse.buttons & Qt.LeftButton) {
                                dragType = 6
                                roi_rect.updateByTop(pos)
                            }
                        } else if (roi_rect.isPointNearLine(pt.x, pt.y, 0, roi_rect.height, roi_rect.width, roi_rect.height, roi_rect._m) && pt.x > 0 && pt.x < roi_rect.width) { // bottom edge
                            roi_rect.setCursorShape(Qt.SizeVerCursor)
                            if (mouse.buttons & Qt.LeftButton) {
                                dragType = 7
                                roi_rect.updateByBottom(pos)
                            }
                        } else {
                            roi_rect.setCursorShape(Qt.SizeAllCursor)
                        }
                    } else {
                        roi_rect.setCursorShape(Qt.ArrowCursor)
                    }
                }
            }

            function setCursorShape(cursorShape) {
                if (roi_rect_mouse.cursorShape !== cursorShape) {
                    roi_rect_mouse.cursorShape = cursorShape
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
                var right = roi_rect.x + roi_rect.width
                var bottom = roi_rect.y + roi_rect.height
                top = Math.min(top, bottom - roi_rect.minimalSize)
                left = Math.min(left, right - roi_rect.minimalSize)
                roi_rect.x = left
                roi_rect.y = top
                roi_rect.width = right - left
                roi_rect.height = bottom - top
            }

            function updateByTopRight(pos) {
                var top = pos.y
                var left = roi_rect.x
                var right = pos.x
                var bottom = roi_rect.y + roi_rect.height
                top = Math.min(top, bottom - roi_rect.minimalSize)
                right = Math.max(right, left + roi_rect.minimalSize)
                roi_rect.x = left
                roi_rect.y = top
                roi_rect.width = right - left
                roi_rect.height = bottom - top
            }

            function updateByBottomLeft(pos) {
                var top = roi_rect.y
                var left = pos.x
                var right = roi_rect.x + roi_rect.width
                var bottom = pos.y
                bottom = Math.max(bottom, top + roi_rect.minimalSize)
                left = Math.min(left, right - roi_rect.minimalSize)
                roi_rect.x = left
                roi_rect.y = top
                roi_rect.width = right - left
                roi_rect.height = bottom - top
            }

            function updateByBottomRight(pos) {
                var top = roi_rect.y
                var left = roi_rect.x
                var right = pos.x
                var bottom = pos.y
                bottom = Math.max(bottom, top + roi_rect.minimalSize)
                right = Math.max(right, left + roi_rect.minimalSize)
                roi_rect.x = left
                roi_rect.y = top
                roi_rect.width = right - left
                roi_rect.height = bottom - top
            }

            function updateByTop(pos) {
                var top = pos.y
                var bottom = roi_rect.y + roi_rect.height
                top = Math.min(top, bottom - roi_rect.minimalSize)
                roi_rect.y = top
                roi_rect.height = bottom - top
            }

            function updateByBottom(pos) {
                var top = roi_rect.y
                var bottom = pos.y
                bottom = Math.max(bottom, top + roi_rect.minimalSize)
                roi_rect.y = top
                roi_rect.height = bottom - top
            }

            function updateByLeft(pos) {
                var left = pos.x
                var right = roi_rect.x + roi_rect.width
                left = Math.min(left, right - roi_rect.minimalSize)
                roi_rect.x = left
                roi_rect.width = right - left
            }

            function updateByRight(pos) {
                var left = roi_rect.x
                var right = pos.x
                right = Math.max(right, left + roi_rect.minimalSize)
                roi_rect.x = left
                roi_rect.width = right - left
            }
        }
    }

    onWidthChanged: {
        if (isFitInView) {
            fitInView()
        }
    }
    onHeightChanged: {
        if (isFitInView) {
            fitInView()
        }
    }

    Rectangle {
        id: drawingRect
        visible: scalableImage.drawing
        color: "transparent"
        border.width: 1
        border.color: scalableImage.drawingColor
    }



    /**
     * @brief 设置图片是否可拖拽
     * @param enable
     */
    function setImageDragEnable(enable) {
        scalableImage.imageDragEnable = enable
    }

    /**
     * @brief 在中心缩放图像
     * @param scale
     */
    function scaleInCenter(scale) {
        // 缩放后的原点
        var scaleOrigin = mapToItem(_image, 0, 0)
        _image.scale = Math.min(Math.max(from, scale), to)
        var dx = (scalableImage.width - _image.sourceSize.width * _image.scale) / 2
        var dy = (scalableImage.height - _image.sourceSize.height * _image.scale) / 2
        var pos = mapFromItem(_image, scaleOrigin)
        // 按照差值移动一下图，使得图看起来在(0,0)处缩放
        _image.x -= pos.x
        _image.y -= pos.y
        // 移动到窗口中央
        _image.x -= scaledImagePos.x - dx
        _image.y -= scaledImagePos.y - dy
        // 不能将上述四条语句合并，因为 x/y 改变时会调用信号槽改变 scaledImagePos
    }

    /**
     * @brief 鼠标滚轮缩放图片, 更新图片位置
     * @param wheel
     */
    function scaleImageByWheel(wheel) {
        // 鼠标相对于缩放前图像的位置
        var scaleOrigin = mapToItem(_image, wheel.x, wheel.y)
        // 缩放
        var step = wheel.angleDelta.y / 120 * scalableImage.stepSize
        // _image.scale = Math.min(Math.max(from * scalableImage.imageSourceScale, _image.scale + step), to * scalableImage.imageSourceScale)
        _image.scale = Math.min(Math.max(from, _image.scale + step), to)
        // 鼠标位置相对于缩放后图像的位置
        var pos = mapFromItem(_image, scaleOrigin)
        //按照差值移动一下图，使得图看起来在鼠标位置缩放
        _image.x -= pos.x - wheel.x
        _image.y -= pos.y - wheel.y
    }

    /**
     * @brief 更新缩放后的图像起始点
     */
    function updateImagePos() {
        scaledImagePos = mapFromItem(_image, 0, 0)
    }

    /**
     * @brief 设置图片区域的鼠标形状
     * @param cursorShape
     */
    function setCursorShape(cursorShape) {
        mouseArea.cursorShape = cursorShape
    }

    /**
     * @brief 图像适应窗口
     */
    function fitInView() {
        if (!scalableImage.isFitInView || _image.sourceSize.height === 0 || _image.sourceSize.width === 0)
            return
        scalableImage.imageSourceScale = Math.min(scalableImage.height / _image.sourceSize.height, scalableImage.width / _image.sourceSize.width)
        // 缩放后的原点
        var scaleOrigin = mapToItem(_image, 0, 0)
        _image.scale = scalableImage.imageSourceScale
        var dx = (scalableImage.width - _image.sourceSize.width * _image.scale) / 2
        var dy = (scalableImage.height - _image.sourceSize.height * _image.scale) / 2
        var pos = mapFromItem(_image, scaleOrigin)
        // 按照差值移动一下图，使得图看起来在(0,0)处缩放
        _image.x -= pos.x
        _image.y -= pos.y
        // 移动到窗口中央
        _image.x -= scaledImagePos.x - dx
        _image.y -= scaledImagePos.y - dy
    }

    /**
     * @brief 开始绘制矩形, 记录起始位置, 重置宽高避免上一次矩形遗留
     * @param mouse
     */
    function startDrawingRect(mouse) {
        scalableImage.startPoint.x = mouse.x
        scalableImage.startPoint.y = mouse.y
        drawingRect.width = 0
        drawingRect.height = 0
        scalableImage.drawing = true
    }

    /**
     * @brief 更新绘制的矩形, 结束绘制
     * @param mouse
     */
    function updateDrawingRectByMouse(mouse) {
        drawingRect.width = Math.abs(mouse.x - scalableImage.startPoint.x)
        drawingRect.height = Math.abs(mouse.y - scalableImage.startPoint.y)
        drawingRect.x = Math.min(mouse.x, scalableImage.startPoint.x)
        drawingRect.y = Math.min(mouse.y, scalableImage.startPoint.y)
    }

    /**
     * @brief 添加一个矩形, 矩形被限制在图像区域内
     * @param mouse
     */
    function updateRect(mouse) {
        var pt1 = mapToItem(_image, scalableImage.startPoint)
        var pt2 = mapToItem(_image, mouse.x, mouse.y)
        var left = Math.min(pt1.x, pt2.x)
        left = Math.max(0, left)
        var right = Math.max(pt1.x, pt2.x)
        right = Math.min(right, _image.width)
        var top = Math.min(pt1.y, pt2.y)
        top = Math.max(0, top)
        var bottom = Math.max(pt1.y, pt2.y)
        bottom = Math.min(bottom, _image.height)
        var x = left
        var y = top
        var width = right - left
        var height = bottom - top
        if (width > 0 && height > 0) {
            // console.log("add one rect", x, y, width, height)
            roi_rect.x = x
            roi_rect.y = y
            roi_rect.width = width
            roi_rect.height = height
        }
    }

    function clearROI() {
        roi_rect.x = 0
        roi_rect.y = 0
        roi_rect.width = 0
        roi_rect.height = 0
        roi_rect.visible = false
    }
}

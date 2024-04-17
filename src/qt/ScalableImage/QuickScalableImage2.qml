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
    property bool opacityEnable: false

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
            scalableImage.opacityEnable = true
            if (mouse.button === Qt.LeftButton) {
                if (mouse.modifiers & Qt.ControlModifier) {
                    setImageDragEnable(true)
                    setCursorShape(Qt.ClosedHandCursor)
                } else {
                    scalableImage.startDrawingRect(mouse)
                }
            } else if (mouse.button === Qt.MiddleButton) {
                setImageDragEnable(true)
                setCursorShape(Qt.ClosedHandCursor)
            }
        }

        onReleased: function (mouse) {
            if (scalableImage.imageDragEnable) {
                setImageDragEnable(false)
                if (mouse.modifiers & Qt.ControlModifier) {
                    setCursorShape(Qt.OpenHandCursor)
                } else {
                    setCursorShape(Qt.ArrowCursor)
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
            scalableImage.opacityEnable = false
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

        Rectangle {
            id: roi_rect
            visible: false
            color: "transparent"
            MouseArea {
                anchors.fill: parent
            }
        }

        layer.enabled: scalableImage.opacityEnable
        layer.effect: OpacityMask {
            maskSource: Rectangle {
                color: Qt.rgba(0,0,0,0.5)
                width: _image.width
                height: _image.height
                Rectangle {
                    x: roi_rect.x
                    y: roi_rect.y
                    width: roi_rect.width
                    height: roi_rect.height
                    color: Qt.rgba(0,0,0,1)
                }
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
            console.log("add one rect", x, y, width, height)
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
    }
}

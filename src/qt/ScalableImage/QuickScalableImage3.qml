import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import ScalableImage as T

Item {
    id: scalableImage
    clip: true
    width: 200
    height: 200

    property var imageRect

    signal updateImageRect
    onUpdateImageRect: {
        var pt1 = mapToItem(_image, 0, 0)
        var pt2 = mapToItem(_image, scalableImage.width, scalableImage.height)
        imageRect = [pt1.x, pt1.y, pt2.x - pt1.x, pt2.y - pt1.y]
    }

    property alias image: _image
    property alias status: _image.status
    property alias source: _image.source
    property alias sourceSize: _image.sourceSize
    property bool imageDragEnable: false
    property real stepSize: {
        if (_image.scale < 2 || _image.paintedWidth * _image.scale < scalableImage.width || _image.paintedHeight * _image.scale < scalableImage.height) {
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
        if (_image.source !== Qt.url("") && _image.status === _image.Ready) {
            return Math.min(scalableImage.height / _image.sourceSize.height, scalableImage.width / _image.sourceSize.width)
        }
        return 1.0
    }

    property point startPoint
    property color drawingColor: "red"
    property bool drawing: false
    property var roiItem: roi_circle

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
                    roiItem.selected = false
                    scalableImage.drawing = true
                    roiItem.startPoint = mapToItem(_image, mouse.x, mouse.y)
                }
            } else if (mouse.button === Qt.MiddleButton) {
                setImageDragEnable(true)
                setCursorShape(Qt.ClosedHandCursor)
                roiItem.setCursorShape(Qt.ClosedHandCursor)
            }
        }

        onReleased: function (mouse) {
            if (scalableImage.imageDragEnable) {
                setImageDragEnable(false)
                if (mouse.modifiers & Qt.ControlModifier) {
                    setCursorShape(Qt.OpenHandCursor)
                } else {
                    setCursorShape(Qt.ArrowCursor)
                    if (roiItem.selected) {
                        roiItem.setCursorShape(Qt.SizeAllCursor)
                    } else {
                        roiItem.setCursorShape(Qt.ArrowCursor)
                    }
                }
            } else if (mouse.button === Qt.LeftButton) {
                if (scalableImage.drawing) {
                    roiItem.updateByPos(mapToItem(_image, mouse.x, mouse.y))
                    scalableImage.drawing = false
                }
            }
        }

        onPositionChanged: function (mouse) {
            if (scalableImage.drawing) {
                roiItem.updateByPos(mapToItem(_image, mouse.x, mouse.y))
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
            getPos()
        } else if (event.key === Qt.Key_Delete) {
            if (roiItem.selected) {
                roiItem.clear()
            }
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
        asynchronous: true
        fillMode: Image.PreserveAspectFit
        onXChanged: {
            updateImagePos()
            scalableImage.updateImageRect()
        }
        onYChanged: {
            updateImagePos()
            scalableImage.updateImageRect()
        }
        onStatusChanged: {
            if (_image.status === Image.Ready) {
                if (isFitInView) {
                    fitInView()
                }
                scalableImage.updateImageRect()
            }
        }
        transformOrigin: Item.TopLeft

        onScaleChanged: {
            scalableImage.updateImageRect()
        }

        // QuickEditableRect {
        //     id: roi_rect
        //     visible: false
        // }

        QuickCircle {
            id: roi_circle
            center.x: 400
            center.y: 400
            radius: 200
            color: "red"
        }
    }

    onWidthChanged: {
        if (isFitInView) {
            fitInView()
        }
        scalableImage.updateImageRect()
    }
    onHeightChanged: {
        if (isFitInView) {
            fitInView()
        }
        scalableImage.updateImageRect()
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
        if (scalableImage.width === 0 || scalableImage.height === 0 || _image.sourceSize.height === 0 || _image.sourceSize.width === 0)
            return
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
        // 不能将上述四条语句合并，因为 image.x/y 改变时会调用信号槽改变 scaledImagePos
    }

    /**
     * @brief 鼠标滚轮缩放图片, 更新图片位置
     * @param wheel
     */
    function scaleImageByWheel(wheel) {
        if (scalableImage.width === 0 || scalableImage.height === 0 || _image.sourceSize.height === 0 || _image.sourceSize.width === 0)
            return      
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
        if (scalableImage.width === 0 || scalableImage.height === 0 || _image.sourceSize.height === 0 || _image.sourceSize.width === 0)
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
}

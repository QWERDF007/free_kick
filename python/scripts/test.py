import time

import win32api
import win32gui

from win32con import WM_INPUTLANGCHANGEREQUEST


def get_language():
    """获取当前输入法状态"""
    hwnd = win32gui.GetForegroundWindow()
    thread_id = win32api.GetWindowLong(hwnd, 0)
    klid = win32api.GetKeyboardLayout(thread_id)
    lid = klid & (2 ** 16 - 1)
    lid_hex = hex(lid)
    if lid_hex == '0x409':
        return 'EN'
    elif lid_hex == '0x804':
        return 'ZH'
    else:
        return 'Unkonwn'
    


def change_language(lang="EN"):
    """
    切换语言
    :param lang: EN--English; ZH--Chinese
    :return: bool
    """
    LANG = {
        "ZH": 0x0804,
        "EN": 0x0409
    }
    hwnd = win32gui.GetForegroundWindow()
    language = LANG[lang]
    result = win32api.SendMessage(
        hwnd,
        WM_INPUTLANGCHANGEREQUEST,
        0,
        language
    )
    if not result:
        return True



if __name__ == '__main__':
    if get_language() == 'ZH':
        change_language('EN')
    time.sleep(5)
    print('当前语言为：', get_language())
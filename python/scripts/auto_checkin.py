import time

import cv2
import pyautogui
import numpy as np

import subprocess

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


URL = "https://cordcloud.biz/"
CHROME = 'C:/Program Files/Google/Chrome/Application/chrome.exe'

LOGIN_TEXT_PIC = './assets/login_text.png'
VERIFY_PIC = './assets/verify.png'
EMAIL_INPUT_PIC = './assets/email_input.png'
PASSWORD_INPUT_PIC = './assets/password_input.png'
LOGIN_BUTTON_PIC = './assets/login_button.png'
CHECKIN_PIC = './assets/checkin.png'
PROFILE_PIC = './assets/profile.png'
GET_RESULT_TEXT_PIC = './assets/get_result_text.png'
CHECKED_PIC = './assets/checked.png'
CONFIRM_RESULT_PIC = './assets/confirm_result.png'

TIMEOUT = 30


EMAIL = ''
PASSWORD = ''



def get_screen_pos_by_template(template_path):
    start = time.time()
    template = cv2.imread(template_path)
    screenshot = np.array(pyautogui.screenshot())
    img = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
    max_val, rect = get_pos_by_matchTemplate(img, template)
    end = time.time()
    # print(f"Time used: {(end-start) * 1000} ms")
    if max_val < 0.9:
        return None, None
    return rect, screenshot

def get_pos_by_matchTemplate(img, template):
    h, w = template.shape[:2]
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    x0, y0 = top_left
    x1, y1 = bottom_right
    return max_val, (x0, y0, x1, y1)

def input_email(email):
    pos, _ = get_screen_pos_by_template(EMAIL_INPUT_PIC)
    if pos is not None:
        pyautogui.click(pos[0]+20, pos[1]+20)
        if get_language() == 'ZH':
            change_language('EN')
        pyautogui.typewrite(email)
        return True
    return False

def input_password(password):
    pos, _ = get_screen_pos_by_template(PASSWORD_INPUT_PIC)
    if pos is not None:
        pyautogui.click(pos[0]+20, pos[1]+20)
        pyautogui.typewrite(password)
        return True
    return False

def check_in():
    pos, _ = get_screen_pos_by_template(CHECKIN_PIC)
    if pos is not None:
        pyautogui.click(pos[0]+30, pos[1]+5)
        time.sleep(1)
        # get_result()
        pyautogui.click(pos[0], pos[1])
        return True
    return False

def confirm_result():
    pos, _ = get_screen_pos_by_template(CONFIRM_RESULT_PIC)
    if pos is not None:
        pyautogui.click(pos[0]+10, pos[1]+5)
        return True
    return False
    
def checked():
    pos, _ = get_screen_pos_by_template(CHECKED_PIC)
    if pos is not None:
        return True
    return False

def go_to_auth_page():
    pos, _ = get_screen_pos_by_template(LOGIN_TEXT_PIC)
    if pos is not None:
        pyautogui.click(pos[0]+8, pos[1]+8)
        return True
    return False

def login():
    pos, _ = get_screen_pos_by_template(LOGIN_BUTTON_PIC)
    if pos is not None:
        pyautogui.click(pos[0]+20, pos[1]+20)
        return True
    return False

def logout():
    pos, _ = get_screen_pos_by_template(PROFILE_PIC)
    if pos is not None:
        pyautogui.click(pos[0]+20, pos[1]+20)
        time.sleep(0.5)
        pyautogui.click(pos[0]-30, pos[1]+140)
        return True
    return False

def verify_you_are_human():
    pos, _ = get_screen_pos_by_template(VERIFY_PIC)
    if pos is not None:
        pyautogui.click(pos[0]+30, pos[1]+30)

def get_result():
    try:
        import pytesseract
        import PIL
        import re
        pos, screenshot = get_screen_pos_by_template(GET_RESULT_TEXT_PIC)
        x0,y0 = pos[0]-10, pos[1]-5
        x1,y1 = pos[2]+120, pos[3]+5
        pimg = PIL.Image.fromarray(cv2.cvtColor(screenshot[y0:y1,x0:x1], cv2.COLOR_BGR2RGB))
        text = pytesseract.image_to_string(pimg)
        pattern = r'\d+MB'
        result = re.search(pattern, text)
        with open('auto_checkin.log', 'a') as f:
            f.write(f"Get result: {result.group()}\n\n")
    except:
        pass




if __name__ == '__main__':
    with open('auto_checkin.log', 'a') as f:
        f.write(f"Start at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
    p = subprocess.Popen([CHROME, URL])
    time.sleep(5)
    start = time.time()
    is_in_home_page = True
    is_in_login_page = False
    is_in_user_page = False
    has_check_in = False
    has_input_email = False
    has_input_password = False
    while True:
        if is_in_home_page and go_to_auth_page():
            is_in_login_page = True
            is_in_home_page = False
        elif is_in_login_page:
            pos = None
            if not has_input_email:
                verify_you_are_human()
            if not has_input_email:
                has_input_email = input_email(EMAIL)
            if not has_input_password:
                has_input_password = input_password(PASSWORD)
            if has_input_email and has_input_password and login():
                is_in_login_page = False
                is_in_home_page = False
                is_in_user_page = True
        elif is_in_user_page:
            if not has_check_in:
                has_check_in = check_in()
            if has_check_in:
                logout()
            if not has_check_in and checked():
                print("Checked")
                if logout():
                    break
        if time.time() - start > TIMEOUT:
            break
        time.sleep(1)
    p.terminate()
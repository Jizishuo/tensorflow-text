from captcha.image import ImageCaptcha
import numpy as np
from PIL import Image
import random
import sys

number = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",\
            "u", "v", "w", "s", "y", "z"]

ALPHABET = []

def random_captcha_text(char_set=number, captcha_size=4):
    #验证码列表
    captcha_text = []
    for i in range(captcha_size):
        #随机选择
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text

#生成字符对应的验证码
def gen_captcha_text_and_image():
    image = ImageCaptcha()
    #获取随机生成的验证码
    captcha_text = random_captcha_text()
    #验证码列表转化为字符串
    captcha_text = "".join(captcha_text)
    #生成验证码
    captcha = image.generate(captcha_text)
    image.write(
        captcha_text,
        "F:\\python项目\\tensorflow--验证码识别\\data-image\\" + captcha_text + ".jpg"
    )

num = 10000
if __name__ == "__main__":
    for i in range(num):
        gen_captcha_text_and_image()
        sys.stdout.write("r\>> Creating image %d/%d" % (i+1, num))
        sys.stdout.flush()
    sys.stdout.write('\n')
    sys.stdout.flush()

    print("完成")
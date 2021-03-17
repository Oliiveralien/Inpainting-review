import base64
from io import BytesIO
from PIL import Image
import requests
import pdb

img = Image.open("4.jpg")
mask = Image.open("4.png")

mode_img = img.mode
mode_msk = mask.mode

W, H = img.size
str_img = img.tobytes().decode("latin1")
str_msk = mask.tobytes().decode("latin1")

data = {'str_img': str_img, 'str_msk': str_msk, 'width':W, 'height':H, 
        'mode_img':mode_img, 'mode_msk':mode_msk}

r = requests.post('http://47.57.135.203:2333/api', json=data)

str_result = r.json()['str_result']

result = str_result.encode("latin1")
result = Image.frombytes('RGB', (W, H), result, 'raw')
result.save("4_result.jpg")

## avoid region (or include)
img = Image.open("0227.jpg")
mask = Image.open("0227_mask.png")
avoid = Image.open("0227_avoid.png")

mode_img = img.mode
mode_msk = mask.mode

W, H = img.size
str_img = img.tobytes().decode("latin1")
str_msk = mask.tobytes().decode("latin1")
str_avoid = avoid.tobytes().decode("latin1")

data = {'str_img': str_img, 'str_msk': str_msk, 'str_include':None, 'str_avoid':str_avoid, 'width':W, 'height':H, 'mode_img':mode_img, 'mode_msk':mode_msk, 'is_refine': True}

r = requests.post('http://47.57.135.203:2333/api', json=data)

str_result = r.json()['str_result']

result = str_result.encode("latin1")
result = Image.frombytes('RGB', (W, H), result, 'raw')
result.save("0227_result.jpg")



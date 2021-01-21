import os
from tqdm import tqdm
import cv2

inp = './load/div2k/DIV2K_valid_HR'
outp = './load/div2k/DIV2K_valid_LR_bicubic/X2'
# print(size)
# os.mkdir(str(size))
filenames = os.listdir(inp)
for filename in tqdm(filenames):
    org_pic = cv2.imread(os.path.join(inp, filename))
    pic = cv2.resize(org_pic, (320, 240), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(outp, filename.split('.')[0] + '.bmp'), pic)


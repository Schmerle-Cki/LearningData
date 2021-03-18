import cv2
import numpy as np
import math
from ProjectWarping import raw_img, modified_img, interpolation
from PIL import Image

MAX_VALUE = 100
epsilon = 1e-5


def sign(x):
    if x >= 0:
        return 1
    else:
        return -1


def trim(x):
    if abs(x) < epsilon:
        return 0
    else:
        return x


# adjust brightness according to LUT
def adjust_per_pixel(source_path, save_img_path, width=128, height=128):
    source = cv2.imread(source_path)
    source_label = np.array(source, dtype='int32')

    width = len(source_label)
    height = len(source_label[0])

    print("width:", str(width), "\theight: ", str(height))

    # padding
    long = width
    delta = int(abs(height - width)/2)
    if height > width:
        long = height
        padding_label = np.zeros([long, long, 3])
        padding_label[:][:] = np.array([130, 130, 130])
        padding_label[delta: long - delta][:][:] = source_label[:][:][:]

    else:
        padding_label = np.zeros([long, long, 3])
        padding_label[:][:] = np.array([130, 130, 130])
        padding_label[:][delta:long-delta][:] = source_label[:][:][:]

    source_label = padding_label

    cx = int(long/2)
    cy = cx

    d0 = 0.5 * max(width, height)
    w_out = width/math.pi
    h_out = height/math.pi
    r0 = min(w_out, h_out)
    r0_int = int(r0)
    print(r0_int*math.pi)

    label = np.empty([2*r0_int + 1, 2*r0_int + 1, 3])

    for i in range(-r0_int, r0_int+1):
        maxj = int(math.sqrt(r0_int**2 - i**2))
        for j in range(-maxj, maxj + 1):
            sign_i = sign(i)
            sign_j = sign(j)
            r = math.sqrt(i**2 + j**2)
            if i == 0:
                theta = math.pi/2
            else:
                theta = math.atan(j/i)
            phi = math.asin(r/r0)
            d = d0*2*phi/math.pi
            y = trim(abs(d*math.sin(theta))*sign_j)
            x = trim(abs(d*math.cos(theta))*sign_i)
            f1 = [int(cx + x), int(cy + y)]
            print(str([r0_int + i, r0_int + j]), str(f1))
            # f2 = trim_position(f1, width, height)
            color = interpolation(f1, [cx+x, cy+y], source_label)
            # color = [i for i in source_label[f2[0]][f2[1]]]
            for channel in range(3):
                label[int(r0_int + i)][int(r0_int + j)][channel] = color[channel]

    cv2.imwrite(save_img_path, label)

    # soften the edge
    img = Image.open(save_img_path)
    img_after = img.resize((img.width, img.height), Image.ANTIALIAS)
    img_after.save(modified_img('afterANTIALIAS.jpg'))
    return label


# image.histogram : RGB concatenated
if __name__ == "__main__":
    adjust_per_pixel(raw_img('warping.png'), modified_img('sphere-building.jpg'))













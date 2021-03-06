from PIL import Image, ImageStat
import cv2
import numpy as np
import os.path
import math
MAX_VALUE = 100


self_LUT = {}
for i in range(0,256):
    self_LUT[i] = i


# limit the output among 0-255
def trim_output(final_brightness):
    if final_brightness <= 0:
        return 0
    if final_brightness >= 255:
        return 255
    return final_brightness


# bias can be negative or positive
def linear_brightness(bias):
    LUT = {}
    for i in range(0, 256):
        LUT[i] = trim_output(i + bias)
    return LUT


def center_contrast(factor):
    LUT = {}
    for i in range(0, 256):
        LUT[i] = trim_output(factor * (i - 127) + 127)
    return LUT


# [min, max]
def contrast_stretch(minI, maxI, minJ, maxJ):
    LUT = {}
    biasI = maxI - minI
    biasJ = maxJ - minJ
    ratio = biasJ/biasI
    for i in range(minI, maxI + 1):
        LUT[i] = trim_output(int(ratio*(i-minI) + minJ))
    return LUT


# increase/decrease lightness naturally
# using library: skimage.exposure.adjust_gamma(image, gamma=1)
def adjust_contrast(gamma):
    LUT = {}
    ratio = 255 / (255**gamma)
    for i in range(0, 256):
        LUT[i] = trim_output(ratio * (i**gamma))
    return LUT


# 直方图：直接调用image.histogram
# compute CDF, input is already normalized
def equalization(histogram):
    degrees = len(histogram)
    LUT = {}
    total = 0
    for i in range(0, degrees):
        total += histogram[i] * degrees
        LUT[i] = trim_output(total)

    return LUT


def matching(source, target):
    total_len = len(source)
    assert total_len == len(target)
    print(total_len)

    LUT = {}
    for i in range(0, total_len):
        bias = 1e10
        best = 0
        for j in range(0, total_len):
            delta = abs(target[j] - source[i])
            if delta < bias:
                best = j
                bias = delta
        LUT[i] = best

    return LUT


# 读取图片原有的亮度值
def brightness(path):
    im = Image.open(path)
    stat = ImageStat.Stat(im)
    r, g, b = stat.mean
    return math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068 *(b**2)) # 0.241 + 0.691 + 0.068 = 1


def get_image_light_gs(dst_src):
    im = Image.open(dst_src)
    stat = ImageStat.Stat(im)
    gs = (math.sqrt(0.241 * (r ** 2) + 0.691 * (g ** 2) + 0.068 * (b ** 2))
          for r, g, b in im.getdata())
    return sum(gs) / stat.count[0]


def mkdir(dir_path):
    folder = os.path.exists(dir_path)

    if not folder:
        os.makedirs(dir_path)
        print("new folder:" + dir_path)
    else:
        print("folder already exists:" + dir_path)


# resize the image
def resize(jpgfile, outdir, width=128,height=128):
    img=Image.open(jpgfile)
    try:
        new_img = img.resize((width, height), Image.BILINEAR)
        new_img.save(os.path.join(outdir, os.path.basename(jpgfile)))
    except Exception as e:
        print(e)


# adjust brightness according to LUT
def adjust_per_pixel(from_img_path, save_img_path, LUT_0, LUT_1, LUT_2, color_type='rgb',array=[], width=128, height=128):
    # resize
    # new_array = cv2.resize(label_im, (width, height), interpolation=cv2.INTER_CUBIC)
    # label = np.array(new_array, dtype='int32')
    if from_img_path != '':
        if color_type == 'rgb':
            label_im = cv2.imread(from_img_path)
        else:
            label_im = cv2.imread(from_img_path, cv2.IMREAD_UNCHANGED)
        label = np.array(label_im, dtype='int32')
    else:
        label = array

    # adjust brightness by pixel
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            if color_type == 'rgb':
                label[i][j][0] = LUT_0[int(label[i][j][0])]
                label[i][j][1] = LUT_1[int(label[i][j][1])]
                label[i][j][2] = LUT_2[int(label[i][j][2])]
            else:
                label[i][j][0] = LUT_0[int(label[i][j][0])]

    cv2.imwrite(save_img_path, label)
    return label


# normalized RGB histogram: separate channel
# the order in open-cv is: B G R
def get_rgb_histogram(img):
    size = img.width * img.height
    histogram = img.histogram()
    histogram = [i/size for i in histogram]
    B = histogram[0: 256]
    G = histogram[256: 512]
    R = histogram[512: 768]
    return histogram, R, G, B


def raw_img(img_name):
    from_base = r'E:\LearningData\Spring\DigitalImageProcessing\ClassicalImage'
    return os.path.join(from_base, img_name)


def modified_img(img_name):
    to_base = r'E:\LearningData\Spring\DigitalImageProcessing\ImageAfter'
    return os.path.join(to_base, img_name)


# image.histogram : RGB concatenated
if __name__ == "__main__":
    dir_path = r'E:\LearningData\Spring\DigitalImageProcessing\ImageAfter'
    mkdir(dir_path)

    # TODO: compare linear and gamma
    LUT_1 = linear_brightness(50)
    LUT_2 = adjust_contrast(0.52)
    adjust_per_pixel(raw_img('greatWall.jpg'), modified_img("greatWall-linear.jpg"), LUT_1, LUT_1, LUT_1)
    adjust_per_pixel(raw_img('greatWall.jpg'), modified_img("greatWall-gamma.jpg"), LUT_2, LUT_2, LUT_2)
    # TODO: brutal brightness
    LUT = linear_brightness(15)
    after_linear = adjust_per_pixel(raw_img('Camera-Man.jpg'), modified_img("Camera-Man-After-linear.jpg"), LUT, LUT,
                                    LUT, color_type='gray')
    LUT = center_contrast(1.5)
    adjust_per_pixel('', modified_img("Camera-Man-After-linear-contrast.jpg"), LUT, LUT, LUT, array=after_linear,
                     color_type='gray')

    # TODO: adjust brightness (gamma)
    LUT = adjust_contrast(gamma=0.95)
    after_brightness = adjust_per_pixel(raw_img('Lena.jpg'), modified_img("Lena-After-gamma.jpg"), LUT, LUT, LUT)

    # TODO: adjust contrast
    LUT = center_contrast(1.5)
    adjust_per_pixel('', modified_img("Lena-After-gamma-contrast.jpg"), LUT, LUT, LUT, array=after_brightness)

    # TODO:balance
    img = Image.open(raw_img('Lena.jpg'))
    h, r, g, b = get_rgb_histogram(img)
    LUT_R = equalization(r)
    LUT_G = equalization(g)
    LUT_B = equalization(b)
    adjust_per_pixel(raw_img('Lena.jpg'), modified_img("Lena-After-balance.jpg"), LUT_R, LUT_G, LUT_B)

    # TODO: stretch -- to a shallower brightness range
    label_im = cv2.imread(raw_img('Camera-Man.jpg'))
    label = np.array(label_im)
    # in case that rgb of each pixel has the same value, only have to pick out the ones globally
    min_1 = label.min()
    max_1 = label.max()
    LUT = contrast_stretch(min_1, max_1, 100, 200)
    adjust_per_pixel(raw_img('Camera-Man.jpg'), modified_img("Camera-Man-After-linear-stretch.jpg"), LUT, LUT, LUT,
                     color_type='gray')

    # TODO: matching
    print("start matching")
    LUT = adjust_contrast(gamma=0.8)
    after_brightness = adjust_per_pixel(raw_img('rainbow.jpg'), modified_img("rainbow-After-gamma-0.8.jpg"), LUT, LUT, LUT)

    LUT = center_contrast(2.0)
    adjust_per_pixel('', modified_img("rainbow-After-gamma-0.8-contrast-2.0.jpg"), LUT, LUT, LUT, array=after_brightness)

    # RGB
    '''from_img = Image.open(modified_img('rainbow-After-gamma-0.8-contrast-2.0.jpg'))
    target_img = Image.open(raw_img('rainbow-cartoon.jpg'))
    h1, r1, g1, b1 = get_rgb_histogram(target_img)
    h2, r2, g2, b2 = get_rgb_histogram(from_img)
    LUT_R = matching(r2, r1)
    LUT_G = matching(g2, g1)
    LUT_B = matching(b2, b1)
    adjust_per_pixel(modified_img('rainbow-After-gamma-0.8-contrast-2.0.jpg'),
                     modified_img('rainbow-After-matching.jpg'), LUT_R, LUT_G, LUT_B)'''

    # grey
    from_img = Image.open(raw_img('chinese_building_grey.jpg'))
    target_img = Image.open(raw_img('Taj_Mahal.jpg'))
    LUT = matching(from_img.histogram()[0: 256], target_img.histogram()[0: 256])
    adjust_per_pixel(raw_img('chinese_building_grey.jpg'),
                     modified_img('chinese-building-After-matching.jpg'), LUT, LUT, LUT)







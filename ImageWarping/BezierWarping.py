import random
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import math
from ProjectWarping import raw_img, modified_img
from PIL import Image
epsilon = 1e-4
resolution = 6000
bias = 10


def trim(x, limitation):
    if x < 0:
        return 0
    if x >= limitation:
        return limitation-1
    return int(x)


def take_first(elem):
    return elem[0]


def take_second(elem):
    return elem[1]


# 4 control points Bezier
class BezierCurve:
    def __init__(self, controls):
        self.points = controls
        self.ratioThird = -1 * controls[0] + 3 * controls[1] - 3 * controls[2] + controls[3]
        self.ratioSecond = 3 * controls[0] - 6 * controls[1] + 3 * controls[2]
        self.ratioFirst = -3 * controls[0] + 3 * controls[1]
        self.ratioZero = controls[0]

        self.dr2 = -3 * controls[0] + 9 * controls[1] - 9 * controls[2] + 3 * controls[3]
        self.dr1 = 6 * controls[0] - 12 * controls[1] + 6 * controls[2]
        self.dr0 = -3 * controls[0] + 3 * controls[1]

        self.miny = controls[3][1]
        self.maxy = controls[1][1]

        self.minx = controls[0][0]
        self.maxx = self.minx

        for i in range(1, 4):
            if controls[i][0] < self.minx:
                self.minx = controls[i][0]
            if controls[i][0] > self.maxx:
                self.maxx = controls[i][0]

    def get_curve_point(self, t):
        return self.ratioZero + self.ratioFirst * t + self.ratioSecond * t**2 + self.ratioThird * t**3

    def get_gradient(self, t):
        return self.dr0 + self.dr1 * t + self.dr2 * t**2

    def from_y_to_x(self, y):
        t = (y - self.miny) / (self.maxy - self.miny)
        epoch = 15
        while epoch > 0:
            if t < 0 or t > 1:
                t = random.random()

            ft = self.get_curve_point(t)[1]
            dft = self.get_gradient(t)[1]

            if abs(ft - y) < epsilon:
                x = self.get_curve_point(t)[0]
                if x >= 0:
                    break
                else:
                    t = random.random()
                    continue

            assert (dft != 0)
            t -= ft / dft
            epoch -= 1
        return self.get_curve_point(t)[0]

    def get_pixel_position(self, point, x_max, width, height, ratio_y):
        # rotate angle: [0, pi]
        if x_max == 0:
            theta = math.pi/2
        else:
            # print(str(point[0]), str(x_max))
            theta = math.acos(point[0]/x_max)
        x = theta/math.pi*width
        y = (point[1]/ratio_y - self.miny)/(self.maxy - self.miny) * height
        print(point, [x,y])
        return [x, y]

    def draw_line(self):
        resolution = 1/600
        points = []
        for i in range(600):
            t = i*resolution
            points.append(self.get_curve_point(t))

        points.sort(key=take_second)
        x_axis = np.linspace(self.minx, self.maxx, 600)
        x = np.array([xx[1] for xx in points])[::3]
        y = np.array([xx[0] for xx in points])[::3]
        # y_axis = interp1d(x, y, kind='cubic')
        plt.plot(x, y)
        plt.show()

    def discrete(self, img_source):
        width = len(img_source)
        height = len(img_source[0])
        label = np.zeros([width, height, 3])
        label[:][:] = [180, 180, 180]

        point_group = []
        step = 1/resolution
        for index in range(resolution):
            t = step * index
            p = self.get_curve_point(t)
            point_group.append(p)

        point_group.sort(key=take_second)
        ratio_y = height/(self.maxy - self.miny)*0.75
        ratio_x = width/2.4/self.maxx
        for i in range(len(point_group)):
            y = int((point_group[i][1] - self.miny) * ratio_y)
            # print("y["+str(i)+"]:"+str(point_group[i]))
            x_max = int(point_group[i][0] * ratio_x)
            for x in range(-x_max, x_max):
                point = self.get_pixel_position([x, y], x_max, width, height, ratio_y)
                label[trim(height - y - bias, height)][trim(x+width/2, width)][:] = img_source[trim(point[0], width)][trim(point[1], height)][:]

        xxx = np.array([xx[0] for xx in point_group])
        yyy = np.array([xx[1] for xx in point_group])
        plt.plot(xxx, yyy)
        plt.show()

        cv2.imwrite(modified_img('curve_warped_building.jpg'), label)


# resize the image
def resize(file_name):
    img = Image.open(raw_img(file_name))
    w = img.width
    h = img.height
    weight = min(w, h)
    try:
        new_img = img.resize((weight, weight), Image.BILINEAR)
        new_img.save(modified_img(file_name))
    except Exception as e:
        print(e)


if __name__ == "__main__":
    points = [np.array([6, 16]), np.array([0, 15.4]), np.array([8, 5]), np.array([1, 3.5])]
    # points = [np.array([3.5, 1]), np.array([5, 8]), np.array([15.4, 0]), np.array([16, 6])]
    line = BezierCurve(points)

    resize('warping.png')

    line.draw_line()

    im = Image.open(modified_img('dragon2.jpg'))
    im.show()

    # 指定逆时针旋转的角度
    im_rotate = im.rotate(-90)
    im_rotate.show()
    im_rotate.save(modified_img('dragon.jpg'))

    # im_rotate.save(modified_img('warping.png'))

    source_img = cv2.imread(modified_img('dragon.jpg'))
    line.discrete(source_img)






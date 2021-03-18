import cv2
import numpy as np
import os.path

MAX_VALUE = 100
epsilon = 1e-5


def trim(number):
    if abs(number) < epsilon:
        return 0
    else:
        return number


def mkdir(dir_path):
    folder = os.path.exists(dir_path)

    if not folder:
        os.makedirs(dir_path)
        print("new folder:" + dir_path)
    else:
        print("folder already exists:" + dir_path)


# limit into picture size
def trim_position(point, width, height):
    res = point
    if point[0] < 0:
        res[0] = 0
    elif point[0] >= width:
        res[0] = width-1

    if point[1] < 0:
        res[1] = 0
    elif point[1] >= height:
        res[1] = height-1

    return res


# TODO: Add Edge Detection
def interpolation(left_down_corner, point, image):
    width = len(image)
    height = len(image[0])
    f1 = trim_position(left_down_corner, width=width, height=height)
    f2 = trim_position([f1[0]+1, f1[1]], width=width, height=height)
    f3 = trim_position([f2[0], f2[1]+1], width=width, height=height)
    f4 = trim_position([f1[0], f1[1]+1], width=width, height=height)

    s = point[0] - f1[0]
    t = point[1] - f1[1]

    # three channels respectively
    final_color = [0, 0, 0]
    for i in range(3):
        final_color[i] = (1 - s) * (1 - t) * image[f1[0]][f1[1]][i] + s * (1 - t) * image[f2[0]][f2[1]][i] + s * t * \
                         image[f3[0]][f3[1]][i] + (1 - s) * t * image[f4[0]][f4[1]][i]
    return final_color


# judge a point whether inside
def is_inside(point, vexes):
    # list four vexes respectively
    va = vexes[0]
    vb = vexes[1]
    vc = vexes[2]
    vd = vexes[3]

    # DOT with each side
    a = (vb[0] - va[0]) * (point[1] - va[1]) - (vb[1] - va[1]) * (point[0] - va[0])
    b = (vc[0] - vb[0]) * (point[1] - vb[1]) - (vc[1] - vb[1]) * (point[0] - vb[0])
    c = (vd[0] - vc[0]) * (point[1] - vc[1]) - (vd[1] - vc[1]) * (point[0] - vc[0])
    d = (va[0] - vd[0]) * (point[1] - vd[1]) - (va[1] - vd[1]) * (point[0] - vd[0])

    if (a >= 0 and b >= 0 and c >= 0 and d >= 0) or (a <= 0 and b <= 0 and c <= 0 and d <= 0):
        return True
    else:
        return False


# adjust brightness according to LUT
def adjust_per_pixel(source_path, target_path, save_img_path, to_points, matrix, width=128, height=128):
    source = cv2.imread(source_path)
    target = cv2.imread(target_path)
    source_label = np.array(source, dtype='int32')
    target_label = np.array(target, dtype='int32')
    label = target_label
    print(label.shape, source_label.shape)

    left_most = min(to_points[0][1], to_points[3][1])
    right_most = max(to_points[1][1], to_points[2][1])
    lower_most = min(to_points[2][0], to_points[3][0])
    upper_most = max(to_points[0][0], to_points[1][0])
    print(left_most, right_most)
    print(lower_most, upper_most)

    # adjust brightness by pixel
    for i in range(lower_most, upper_most + 1):
        for j in range(left_most, right_most + 1):
            point = [i, j]
            if is_inside(point, to_points):
                print("inside", str(point))
                warp_point = get_pixel_position(matrix, point)
                # get target color
                f1 = [int(warp_point[0]), int(warp_point[1])]
                color = interpolation(f1, warp_point, source_label)
                # print(label[i][j])
                for color_channel in range(3):
                    label[i][j][color_channel] = color[color_channel]
                # print(label[i][j])

    cv2.imwrite(save_img_path, label)
    return label


# specified position for translated pixel
def get_pixel_position(matrix, point):
    point = np.mat([[point[0]], [point[1]], [1]])
    result = np.dot(matrix, point)
    factor = float(result[2])
    if abs(factor) < epsilon:
        factor = epsilon
    return [float(result[0])/factor, float(result[1])/factor]


# return the inverse transform matrix (set i to 1)
def get_project_warping_matrix(target_points, source_points):
    # factors by source & target points
    inverse_translation = []
    solution = []
    for i in range(4):
        to_point = target_points[i]
        from_point = source_points[i]
        x = [to_point[0], to_point[1], 1, 0, 0, 0, -from_point[0]*to_point[0], -from_point[0]*to_point[1]]
        y = [0, 0, 0, to_point[0], to_point[1], 1, -from_point[1]*to_point[0], -from_point[1]*to_point[1]]
        solution.append([from_point[0]])
        solution.append([from_point[1]])
        inverse_translation.append(x)
        inverse_translation.append(y)

    inverse_translation = np.mat(inverse_translation)
    solution = np.mat(solution)
    matrix = np.linalg.solve(inverse_translation, solution)

    # convert one-dimension to 3*3 matrix
    inverse = []
    for i in range(2):
        row = []
        for j in range(3):
            index = i * 3 + j
            row.append(trim(float(matrix[index])))
        inverse.append(row)

    row = [trim(float(matrix[6])), trim(float(matrix[7])), 1]
    inverse.append(row)
    return np.mat(inverse)


def get_affine_warping_matrix(target_points, source_points):
    inverse_translation = []
    solution = []
    for i in range(3):
        to_point = target_points[i]
        from_point = source_points[i]
        x = [to_point[0], to_point[1], 1, 0, 0, 0]
        y = [0, 0, 0, to_point[0], to_point[1], 1]
        solution.append([from_point[0]])
        solution.append([from_point[1]])
        inverse_translation.append(x)
        inverse_translation.append(y)

    inverse_translation = np.mat(inverse_translation)
    solution = np.mat(solution)
    matrix = np.linalg.solve(inverse_translation, solution)

    # convert one-dimension to 3*3 matrix
    inverse = []
    for i in range(2):
        row = []
        for j in range(3):
            index = i * 3 + j
            row.append(trim(float(matrix[index])))
        inverse.append(row)

    inverse.append([0, 0, 1])
    return np.mat(inverse)


def raw_img(img_name):
    from_base = r'E:\LearningData\Spring\DigitalImageProcessing'
    return os.path.join(from_base, img_name)


def modified_img(img_name):
    to_base = r'E:\LearningData\LearningData\ImageWarping\AfterWarped'
    return os.path.join(to_base, img_name)


# image.histogram : RGB concatenated
if __name__ == "__main__":
    dir_path = r'E:\LearningData\LearningData\ImageWarping\AfterWarped'
    mkdir(dir_path)

    to = [[314, 171], [387, 507], [265, 534], [196, 194]]
    From = [[523, 0], [523, 698], [0, 698], [0, 0]]

    project_matrix = get_project_warping_matrix(to, From)
    adjust_per_pixel(raw_img('source.jpg'), raw_img('target.jpg'), modified_img('project_warped.jpg'), to, project_matrix)
    affine_matrix = get_affine_warping_matrix(to[0:-1], From[0:-1])
    adjust_per_pixel(raw_img('source.jpg'), raw_img('target.jpg'), modified_img('affine_warped.jpg'), to, affine_matrix)













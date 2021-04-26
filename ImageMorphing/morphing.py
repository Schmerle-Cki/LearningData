import numpy as np
import cv2
import matplotlib.pyplot as plt
from triangle import from_img_point_to_mesh, load_tri_array
from scipy.spatial import Delaunay
MAX_VALUE = 100
epsilon = 1e-5


def center(vertex):
    return ((vertex[0] + vertex[1] + vertex[2])/3).astype(int)


def trim(number):
    if abs(number) < epsilon:
        return 0
    else:
        return number


# Inverse
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

    # TODO: convert one-dimension to 3*3 matrix
    inverse = []
    for i in range(2):
        row = []
        for j in range(3):
            index = i * 3 + j
            row.append(trim(float(matrix[index])))
        inverse.append(row)

    inverse.append([0, 0, 1])
    return np.mat(inverse)


# judge a point whether inside
def is_inside(point, vexes):
    # list 3 vexes respectively
    va = vexes[0]
    vb = vexes[1]
    vc = vexes[2]

    # DOT with each side
    a = (vb[0] - va[0]) * (point[1] - va[1]) - (vb[1] - va[1]) * (point[0] - va[0])
    b = (vc[0] - vb[0]) * (point[1] - vb[1]) - (vc[1] - vb[1]) * (point[0] - vb[0])
    c = (va[0] - vc[0]) * (point[1] - vc[1]) - (va[1] - vc[1]) * (point[0] - vc[0])

    if (a >= 0 and b >= 0 and c >= 0) or (a <= 0 and b <= 0 and c <= 0):
        return True
    else:
        return False


# specified position for translated pixel
def get_pixel_position(matrix, point):
    point = np.mat([[point[0]], [point[1]], [1]])
    result = np.dot(matrix, point)
    '''factor = float(result[2])
    if abs(factor) < epsilon:
        factor = epsilon'''
    factor = 1
    return [float(result[0])/factor, float(result[1])/factor]


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
# TODO: left_corner、point has 'x' horizontally, while image has 'x' vertically
def interpolation(left_down_corner, point, image):
    width = len(image)
    height = len(image[0])

    left_down_corner = [left_down_corner[1], left_down_corner[0]]
    point = [point[1], point[0]]

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


def bounding(tri_vex):
    x_min = tri_vex[0][0]
    x_max = tri_vex[1][0]
    if tri_vex[1][0] < x_min:
        x_min = tri_vex[1][0]
        x_max = tri_vex[0][0]
    if tri_vex[2][0] < x_min:
        x_min = tri_vex[2][0]
    if tri_vex[2][0] > x_max:
        x_max = tri_vex[2][0]

    y_min = tri_vex[0][1]
    y_max = tri_vex[1][1]
    if tri_vex[1][1] < y_min:
        y_min = tri_vex[1][1]
        y_max = tri_vex[0][1]
    if tri_vex[2][1] < y_min:
        y_min = tri_vex[2][1]
    if tri_vex[2][1] > y_max:
        y_max = tri_vex[2][1]

    return [x_min, y_min], [x_max, y_max]


# TODO: points in vertex: 'x' is horizontal
def get_middle_pixel_color(vertex, matrix, label, ret_label):
    edge_min, edge_max = bounding(vertex)
    # TODO: in img label, j is horizontal, k is vertical
    for j in range(edge_min[0], edge_max[0] + 1):
        for k in range(edge_min[1], edge_max[1] + 1):
            point = [j, k]
            if is_inside(point, vertex):
                # TODO: matrix dot calculate
                warp_point = get_pixel_position(matrix, point)
                # TODO: get target color
                f1 = [int(warp_point[0]), int(warp_point[1])]
                color = interpolation(f1, warp_point, label)

                for color_channel in range(3):
                    ret_label[k][j][color_channel] = color[color_channel]


def get_pixel_color(point, matrix, label):
    warp_point = get_pixel_position(matrix, point)
    # TODO: Left_bottom_corner
    f1 = [int(warp_point[0]), int(warp_point[1])]
    color = np.array(interpolation(f1, warp_point, label))
    return color


# TODO: mesh is formed by feature points
# TODO: 'x' is horizontal, 'y' is vertical
def main(t, source_mesh, target_mesh, src_png, trg_png, pid=1, base_dir=r'./face morphing/', middle_write=True):
    # TODO: img label array: 'x' is **vertical**!
    source = np.array(cv2.imread(base_dir + src_png), dtype='int32')
    target = np.array(cv2.imread(base_dir + trg_png), dtype='int32')

    # TODO: label for draw mesh lines
    source_for_mesh = source.copy()
    target_for_mesh = target.copy()

    s_shape = source.shape
    t_shape = target.shape
    shape_0 = min(s_shape[0], t_shape[0])
    shape_1 = min(s_shape[1], t_shape[1])

    # TODO: delaunay is computed by weighted mixture
    delaunay_0 = from_img_point_to_mesh(source_mesh, target_mesh, t)
    delaunay = delaunay_0.simplices.copy()

    matrix_group_m_s = []
    matrix_group_m_t = []

    # TODO: calculate all affine matrix for each triangle at once
    for i in range(delaunay.shape[0]):
        delta_index = delaunay[i]
        one = delta_index[0]
        two = delta_index[1]
        thr = delta_index[2]

        # TODO: Find Triangle Vertex, 'x' is horizontal
        source_vertex = np.array([source_mesh[one], source_mesh[two], source_mesh[thr]])
        target_vertex = np.array([target_mesh[one], target_mesh[two], target_mesh[thr]])

        # TODO: Draw mesh lines and note the triangle index__just for process debug
        cv2.polylines(source_for_mesh, [source_vertex], True, (0, 0, 255))
        cv2.polylines(target_for_mesh, [target_vertex], True, (0, 0, 255))
        s_center = center(source_vertex)
        t_center = center(target_vertex)
        cv2.putText(source_for_mesh, str(i), (s_center[0], s_center[1]), cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 0, 0), 0)
        cv2.putText(target_for_mesh, str(i), (t_center[0], t_center[1]), cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 0, 0), 0)

        # TODO: get affine matrix respectively -- inverse warping
        middle_vertex = (1-t) * source_vertex + t * target_vertex
        middle_vertex = middle_vertex.astype(int)

        matrix_s_m = cv2.getAffineTransform(np.float32(middle_vertex), np.float32(source_vertex))
        matrix_group_m_s.append(matrix_s_m)

        matrix_t_m = cv2.getAffineTransform(np.float32(middle_vertex), np.float32(target_vertex))
        matrix_group_m_t.append(matrix_t_m)

    print('=====================MATRIX CALC DONE======================')

    # TODO: label for affine warping
    middle_label_s = source.copy()
    middle_label_t = target.copy()
    final = np.random.randn(shape_0, shape_1, 3)
    # TODO: for each pixel in final label, inverse warping__find color in s and t
    for i in range(shape_0):
        for j in range(shape_1):
            mesh_id = delaunay_0.find_simplex([j, i])
            matrix_s_m = matrix_group_m_s[mesh_id]
            matrix_t_m = matrix_group_m_t[mesh_id]

            # TODO: when get matrix, the order is (shape[1], shape[0])
            color_s = get_pixel_color([j, i], matrix_s_m, source)
            color_t = get_pixel_color([j, i], matrix_t_m, target)

            middle_label_s[i][j][:] = color_s[:]
            middle_label_t[i][j][:] = color_t[:]
            for color_id in range(3):
                final[i][j][color_id] = t * color_t[color_id] + (1-t) * color_s[color_id]

    if middle_write:
        cv2.imwrite(base_dir + 'source_' + str(pid) + '_mesh_t_' + str(t) + '.jpg', source_for_mesh)
        cv2.imwrite(base_dir + 'target_' + str(pid) + '_mesh_t_' + str(t) + '.jpg', target_for_mesh)
        cv2.imwrite(base_dir + 'source_' + str(pid) + '_m_t_' + str(t) + '.jpg', middle_label_s)
        cv2.imwrite(base_dir + 'target_' + str(pid) + '_m_t_' + str(t) + '.jpg', middle_label_t)

    # TODO: code unused: traverse the triangles and judge whether a point is in __ not accurate, small mesh
    '''
    # TODO: label for affine warping
    middle_label_s = source.copy()
    middle_label_t = target.copy()
    for i in range(delaunay.shape[0]):
        delta_index = delaunay[i]
        one = delta_index[0]
        two = delta_index[1]
        thr = delta_index[2]

        # TODO: Find Triangle Vertex, 'x' is horizontal
        source_vertex = np.array([source_mesh[one], source_mesh[two], source_mesh[thr]])
        target_vertex = np.array([target_mesh[one], target_mesh[two], target_mesh[thr]])

        # TODO: Draw mesh lines and note the triangle index
        cv2.polylines(source_for_mesh, [source_vertex], True, (0, 0, 255))
        cv2.polylines(target_for_mesh, [target_vertex], True, (0, 0, 255))
        s_center = center(source_vertex)
        t_center = center(target_vertex)
        cv2.putText(source_for_mesh, str(i), (s_center[0], s_center[1]), cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 0, 0), 0)
        cv2.putText(target_for_mesh, str(i), (t_center[0], t_center[1]), cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 0, 0), 0)

        middle_vertex = t*source_vertex + (1-t)*target_vertex
        middle_vertex = middle_vertex.astype(int)

        # TODO: get affine matrix respectively -- inverse warping
        # matrix_s_m = get_affine_warping_matrix(middle_vertex, source_vertex)
        matrix_s_m = cv2.getAffineTransform(np.float32(middle_vertex), np.float32(source_vertex))
        get_middle_pixel_color(source_vertex, matrix_s_m, source, middle_label_s)

        matrix_t_m = cv2.getAffineTransform(np.float32(middle_vertex), np.float32(target_vertex))
        # matrix_t_m = get_affine_warping_matrix(middle_vertex, target_vertex)
        get_middle_pixel_color(target_vertex, matrix_t_m, target, middle_label_t)


    final = np.random.randn(shape_0, shape_1, 3)
    final = np.array(final)
    for i in range(shape_0):
        for j in range(shape_1):
            for colorID in range(3):
                final[i][j][colorID] = float(t)*middle_label_s[i][j][colorID] + float(1-t)*middle_label_t[i][j][colorID]
    '''

    print('********************* before write ***********************')
    cv2.imwrite(base_dir + str(pid) + '_final_t_' + str(t) + '.jpg', final)


if __name__ == "__main__":
    for i in range(1, 10):
        source_mesh = np.array(load_tri_array('source1.png'))
        target_mesh = np.array(load_tri_array('target1.png'))
        main(float('%0.1f' % (i * 0.1)), source_mesh, target_mesh, 'source1.png', 'target1.png', 1)

    for i in range(1, 10):
        source_mesh = np.array(load_tri_array('source2.png'))
        target_mesh = np.array(load_tri_array('target2.png'))
        main(float('%0.1f' % (i * 0.1)), source_mesh, target_mesh, 'source2.png', 'target2.png', 2)




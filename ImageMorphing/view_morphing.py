import numpy as np
from morphing import trim, epsilon, main
import math
from calc import rotate_mat
from triangle import load_tri_array
from hand_label import label_one_img
import cv2
import json

delta = 2


def pole(matrix):
    eigenvalue, feature_vector = np.linalg.eig(matrix)

    min_index = np.argmin(eigenvalue)
    min_value = eigenvalue[min_index]
    vector = feature_vector[:, min_index]
    return min_value, vector


def calc_theta(e, d):
    theta = np.arctan2(e[2], (d[1] * e[0] - d[0] * e[1]))
    return theta


def calc_phi(e):
    if e[0] == 0:
        e[0] = epsilon
    return -math.atan(e[1]/e[0])


def pre_warping(pid):
    src_png = 'source_' + str(pid) + '.png'
    trg_png = 'target_' + str(pid) + '.png'
    src_p = np.array(load_tri_array(src_png, r'./view morphing/'))
    trg_p = np.array(load_tri_array(trg_png, r'./view morphing/'))
    # TODO: test 2 is vertical
    if pid == 2:
        src_p = np.array([[src_p[i][1], src_p[i][0]] for i in range(len(src_p))])
        trg_p = np.array([[trg_p[i][1], trg_p[i][0]] for i in range(len(trg_p))])

    src_p_1 = src_p.copy()
    trg_p_1 = trg_p.copy()
    # TODO: remove the outer frame for face
    src_p = src_p[: -8]
    trg_p = trg_p[: -8]

    # TODO: label 8 points for source and target respectively
    # F = calc_f(src_p, trg_p)
    F, mask = cv2.findFundamentalMat(src_p, trg_p)#, cv2.FM_8POINT)
    F_T = F.T
    v0, e0 = pole(F)
    v1, e1 = pole(F_T)

    # TODO: select axis for I0 & I1
    d0 = np.array([-e0[1], e0[0], 0])
    d1 = np.dot(F, d0)
    d1 = np.array([-d1[1], d1[0], 0])

    # TODO: calc rotate theta for I0 & I1
    theta0 = calc_theta(e0, d0)
    theta1 = calc_theta(e1, d1)

    # TODO: calc rotate matrix to XoY
    rotate0 = rotate_mat(d0, theta0)
    rotate1 = rotate_mat(d1, theta1)

    # TODO: calc rotate matrix to Z, eni[3] should be close to 0
    en0 = np.dot(rotate0, e0)
    en1 = np.dot(rotate1, e1)
    # print('====================new epipole==============:', str(en0), str(en1))

    phi0 = calc_phi(en0)
    phi1 = calc_phi(en1)
    rotate0_n = np.array([[np.cos(phi0), -np.sin(phi0), 0],
                          [np.sin(phi0), np.cos(phi0), 0],
                          [0, 0, 1]])
    rotate1_n = np.array([[np.cos(phi1), -np.sin(phi1), 0],
                          [np.sin(phi1), np.cos(phi1), 0],
                          [0, 0, 1]])

    # TODO: calc matrix T
    '''F_head_0 = np.dot(rotate1_n, rotate1)
    F_head_1 = np.dot(rotate_mat(d0, -theta0), rotate_mat([0, 0, 1], -phi0))
    F_head = np.dot(F_head_0, F)
    F_head = np.do(F_head, F_head_1)'''
    H0 = np.dot(rotate0_n, rotate0)
    H1 = np.dot(rotate1_n, rotate1)

    return H0, H1, src_p_1, trg_p_1


def swap_x_y(matrix):
    shape = matrix.shape
    ret = np.zeros([shape[1], shape[0], 3])
    for i in range(shape[0]):
        for j in range(shape[1]):
            ret[j][i][:] = matrix[i][j][:]
    return ret


def adjust_border(corner, horizontal, vertical):
    if corner[0][0] < 0:
        if corner[0][1] < 0:
            corner[0] = [0, 0]
    if corner[1][0] < 0 or corner[1][1] >= vertical:
        corner[1] = [0, vertical-1]
    if corner[2][0] >= horizontal or corner[2][1] < 0:
        corner[2] = [horizontal-1, 0]
    if corner[3][0] >= horizontal or corner[3][1] >= vertical:
        corner[3] = [horizontal-1, vertical-1]

    return corner


def view_morphing(pid):
    H0, H1, src_points, trg_points = pre_warping(pid)
    src_points_list = []
    trg_points_list = []
    for i in range(src_points.shape[0]):
        src_points_list.append([src_points[i][0], src_points[i][1]])
        trg_points_list.append([trg_points[i][0], trg_points[i][1]])
    # print(H0， H1)

    source = np.array(cv2.imread(r'./view morphing/source_' + str(pid) + '.png'))
    target = np.array(cv2.imread(r'./view morphing/target_' + str(pid) + '.png'))
    s_feature_after = []
    t_feature_after = []
    if pid == 2:
        source = swap_x_y(source)
        target = swap_x_y(target)

    s0 = source.shape[0]
    s1 = source.shape[1]

    extend = int(math.sqrt(source.shape[0] ** 2 + source.shape[1] ** 2))
    if pid == 1:
        s_pre = cv2.warpPerspective(source / 255, H0, (s1, s0))
        t_pre = cv2.warpPerspective(target / 255, H1, (s1, s0))
    else:
        s_p = []
        t_p = []

        Hr0 = np.linalg.inv(H0)
        Hr1 = np.linalg.inv(H1)

        for i in range(s0):
            for j in range(s1):
                p = np.dot(H0, [j, i, 1])
                if p[2] == 0:
                    p[2] = epsilon
                s_p.append([p[0]/p[2], p[1]/p[2]])
                q = np.dot(H1, [j, i, 1])
                if q[2] == 0:
                    q[2] = epsilon
                t_p.append([q[0]/q[2], q[1]/q[2]])

        s_p = np.array(s_p)
        t_p = np.array(t_p)

        s_max_x = max(abs(np.max(s_p[:, 0])), abs(np.min(s_p[:, 0])))
        s_max_y = max(abs(np.max(s_p[:, 1])), abs(np.min(s_p[:, 1])))
        t_max_x = max(abs(np.max(t_p[:, 0])), abs(np.min(t_p[:, 0])))
        t_max_y = max(abs(np.max(t_p[:, 1])), abs(np.min(t_p[:, 1])))
        max_x = int(max(s_max_x, t_max_x)) + 10
        max_y = int(max(s_max_y, t_max_y)) + 10

        ss0 = 2*max_y
        ss1 = 2*max_x
        print(ss0, ss1)

        s_pre = np.zeros([ss0, ss1, 3])
        t_pre = np.zeros([ss0, ss1, 3])
        s_f_id = []
        t_f_id = []
        s_f_tmp = []
        t_f_tmp = []
        for i in range(-max_y, max_y):
            for j in range(-max_x, max_x):
                p = np.dot(Hr0, [j, i, 1])
                if p[2] == 0:
                    p[2] = epsilon
                p = p/p[2]
                if 0 <= p[0] < s1 and 0 <= p[1] < s0:
                    s_pre[i + max_y][j + max_x][:] = source[int(p[1])][int(p[0])][:] / 255
                    try:
                        index = src_points_list.index([int(p[0]), int(p[1])])
                        s_f_id.append(index)
                        s_f_tmp.append([j + max_x, i + max_y])
                        # cv2.circle(s_pre, (j + max_x, i + max_y), 2, (0, 255, 0), 1)
                    except:
                        pass

                q = np.dot(Hr1, [j, i, 1])
                if q[2] == 0:
                    q[2] = epsilon
                q = q/q[2]
                if 0 <= q[0] < s1 and 0 <= q[1] < s0:
                    t_pre[i + max_y][j + max_x][:] = target[int(q[1])][int(q[0])][:] / 255
                    try:
                        index = trg_points_list.index([int(q[0]), int(q[1])])
                        t_f_id.append(index)
                        t_f_tmp.append([j + max_x, i + max_y])
                        # cv2.circle(t_pre, (j + max_x, i + max_y), 2, (0, 255, 0), 1)
                    except:
                        pass
                # TODO: add edge points: forward warping
                for h in range(1, 9):
                    if abs(p[0] - src_points[-h][0]) < delta and abs(p[1] - src_points[-h][1]) < delta:
                        s_f_id.append(-h)
                        s_f_tmp.append([j + max_x, i + max_y])
                    if abs(q[0] - trg_points[-h][0]) < delta and abs(q[1] - trg_points[-h][1]) < delta:
                        t_f_id.append(-h)
                        t_f_tmp.append([j + max_x, i + max_y])

        # TODO: merge the warped legal feature points
        for f_id in range(len(s_f_id)):
            try:
                t_id = t_f_id.index(s_f_id[f_id])
                s_feature_after.append(s_f_tmp[f_id])
                # cv2.circle(s_pre, (s_f_tmp[f_id][0], s_f_tmp[f_id][1]), 2, (0, 255, 0), 1)
                t_feature_after.append(t_f_tmp[t_id])
                # cv2.circle(t_pre, (t_f_tmp[t_id][0], t_f_tmp[t_id][1]), 2, (0, 255, 0), 1)
            except:
                pass

        '''
        s_pp = np.zeros([ss0, max_x, 3])
        t_pp = np.zeros([ss0, max_x, 3])
        for i in range(ss0):
            for j in range(max_x):
                s_pp[i][j][:] = s_pre[i][j + max_x][:]
                t_pp[i][j][:] = t_pre[i][j + max_x][:]
        s_pre = s_pp
        t_pre = t_pp
        '''

    cv2.imwrite(r'./view morphing/s_' + str(pid) + '_pre.png', (np.clip(s_pre, 0., 1.) * 255).astype(np.uint8))
    cv2.imwrite(r'./view morphing/t_' + str(pid) + '_pre.png', (np.clip(t_pre, 0., 1.) * 255).astype(np.uint8))

    # TODO: calc position for feature points after rotate
    # TODO: for test 2, features also needs adjust
    if pid == 1:
        for pt in src_points:
            pt_after = np.dot(H0, np.array([pt[0], pt[1], 1]))
            s_feature_after.append([pt_after[0] / pt_after[2], pt_after[1] / pt_after[2]])

        for pt in trg_points:
            pt_after = np.dot(H1, np.array([pt[0], pt[1], 1]))
            t_feature_after.append([pt_after[0] / pt_after[2], pt_after[1] / pt_after[2]])

    json.dump({'data': s_feature_after}, open(r'./view morphing/s_' + str(pid) + '_pre.png_array.json', 'w'))
    json.dump({'data': t_feature_after}, open(r'./view morphing/t_' + str(pid) + '_pre.png_array.json', 'w'))

    s_feature_after = np.array(s_feature_after, dtype='int32')
    t_feature_after = np.array(t_feature_after, dtype='int32')

    # TODO: calculate border points in morphed image: should be (y, x)
    '''
    border = [[0, 0], [0, s0], [s1, 0], [s1, s0]]
    s_border = []
    t_border = []
    for i in range(4):
        s_border.append(np.dot(H0, np.array([border[i][0], border[i][1], 1])))
        t_border.append(np.dot(H1, np.array([border[i][0], border[i][1], 1])))
    # TODO: s_border, t_border is still in the form of (y,x)
    s_border = np.array([[pt[0] / pt[2], pt[1] / pt[2]] for pt in s_border])
    t_border = np.array([[pt[0] / pt[2], pt[1] / pt[2]] for pt in t_border])
    print(s_border)
    print(t_border)
    m_border = (1-t) * s_border + t * t_border
    border = np.array(border)
    Hs, ignore = cv2.findHomography(m_border, border)
    '''

    # TODO: morph
    for i in range(1, 10):
        t = float('%0.1f' % (i * 0.1))
        main(t, s_feature_after, t_feature_after, 's_' + str(pid) + '_pre.png', 't_' + str(pid) + '_pre.png', pid=pid,
             base_dir='./view morphing/', middle_write=True)
        final_b = np.array(cv2.imread(r'./view morphing/' + str(pid) + '_final_t_' + str(t) + '.jpg'), dtype='int32')
        final_b = final_b / 255
        # TODO: manually select the matrix Hs
        a, b = label_one_img(r'./view morphing/' + str(pid) + '_final_t_' + str(t) + '.jpg')
        source_points = np.array([[a[k], b[k]] for k in range(4)]).astype('float32')
        target_points = np.array([[0, 0], [0, s0], [s1, 0], [s1, s0]]).astype('float32')
        print(source_points)
        print(target_points)
        Hs = cv2.getPerspectiveTransform(source_points, target_points)

        final = cv2.warpPerspective(final_b, Hs, (s1, s0), borderMode=cv2.BORDER_REPLICATE)
        if pid == 2:
            final = swap_x_y(final)
        cv2.imwrite(r'./view morphing/warped_' + str(pid) + '_' + str(t) + '.png',
                    (np.clip(final, 0., 1.) * 255).astype(np.uint8))


if __name__ == '__main__':
    view_morphing(1)
    view_morphing(2)








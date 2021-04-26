from scipy import sparse
import numpy as np
import cv2
import json


def load_inner_and_edge(pid=1):
    file_name = r'./image fusion/test' + str(pid) + '_points.json'
    points_dict = dict(json.load(open(file_name, 'r')))
    return points_dict['edge'], points_dict['inner']


def formal_grad(img, points):
    grads = []
    for pt in points:
        grad = img[pt[0]-1][pt[1]][:] + img[pt[0]+1][pt[1]][:] + img[pt[0]][pt[1]-1][:] \
               + img[pt[0]][pt[1]+1][:] - 4*img[pt[0]][pt[1]][:]
        grads.append(grad)
    return grads


def is_neighbor(me, she):
    if me[0] == she[0] and abs(she[1] - me[1]) == 1:
        return True
    if me[1] == she[1] and abs(she[0] - me[0]) == 1:
        return True
    return False


# TODO: edge, inner 'x' is already vertical
def fusion(pid, post, offset):
    edge, inner = load_inner_and_edge(pid)
    # edge = np.array(edge)
    # inner = np.array(inner)
    inner_size = len(inner)
    edge_size = len(edge)
    masked_img = np.array(cv2.imread(r'./image fusion/test' + str(pid) + '_src_ok' + post), dtype='int32')/255
    origin_img = np.array(cv2.imread(r'./image fusion/test' + str(pid) + '_target' + post), dtype='int32')/255
    brutal_img = np.array(cv2.imread(r'./image fusion/test' + str(pid) + '_target_brutal' + post), dtype='int32')/255

    original_grads = formal_grad(masked_img, inner)
    # TODO: A is Laplacian kernel, b is div(g) - sum(edge(neighbor))
    A = np.zeros((inner_size + edge_size, inner_size + edge_size))
    b = np.zeros((inner_size + edge_size, 3))
    # TODO: edge -- keep the same
    for pt_id in range(edge_size):
        id_in_matrix = pt_id + inner_size
        A[id_in_matrix][id_in_matrix] = 1
        [x, y] = edge[pt_id]
        b[id_in_matrix][:] = origin_img[x + offset[0]][y + offset[1]][:]

    # TODO: inner points: neighbors; b = original divergence
    for pt_id in range(inner_size):
        A[pt_id][pt_id] = - 4
        pt = inner[pt_id]

        left = [pt[0] - 1, pt[1]]
        right = [pt[0] + 1, pt[1]]
        up = [pt[0], pt[1] - 1]
        down = [pt[0], pt[1] + 1]
        neighbor = [left, right, up, down]

        for nb in neighbor:
            try:
                index = inner.index(nb)
                A[pt_id][index] = 1
            except:
                try:
                    index = edge.index(nb)
                    A[pt_id][index + inner_size] = 1
                except:
                    pass

        b[pt_id][:] = original_grads[pt_id][:]

    A = np.mat(A)
    x = np.zeros((inner_size + edge_size, 3))
    for color_channel in range(3):
        bb = b[:, color_channel]
        xx = np.linalg.solve(A, bb)
        x[:, color_channel] = xx

    max_color = np.max(x)
    print(max_color)

    for pt in edge:
        brutal_img[pt[0] + offset[0]][pt[1] + offset[1]][:] = origin_img[pt[0] + offset[0]][pt[1] + offset[1]][:]
    for pt_id in range(inner_size):
        pt = inner[pt_id]
        brutal_img[pt[0] + offset[0]][pt[1] + offset[1]][:] = np.array(x[pt_id][:])

    cv2.imwrite(r'./image fusion/fusion_' + str(pid) + '.jpg', (np.clip(brutal_img, 0., 1.) * 255).astype(np.uint8))
    print('===================WRITE OK====================')


if __name__ == '__main__':
    fusion(1, '.jpg', [0, 0])
    fusion(2, '.png', [165, 140])



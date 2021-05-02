import cv2
import numpy as np
from Quantization import Table_8, JPEG, Canon, Nikon
block_size = 8
rate = 1/4


# TODO： transform matrix to zig-1D --- return point order
def zig_1d(shape):
    start_point = [0, 0]
    direction = 0
    flag = 0

    point_count = shape[0] * shape[1]
    array = [start_point]
    while len(array) < point_count:
        if direction == 0:
            if flag == 0:
                start_point = [start_point[0], start_point[1] + 1]
            else:
                start_point = [start_point[0] + 1, start_point[1]]
            [x, y] = start_point
            while x < shape[0] and y >= 0:
                array.append([x, y])
                x += 1
                y -= 1
        else:
            if flag == 0:
                start_point = [start_point[0] + 1, start_point[1]]
            else:
                start_point = [start_point[0], start_point[1] + 1]
            [x, y] = start_point
            while x >= 0 and y < shape[1]:
                array.append([x, y])
                x -= 1
                y += 1

        start_point = array[-1]
        direction = int(1 - direction)
        if start_point == [shape[0] - 1, 0] or start_point == [0, shape[1] - 1]:
            flag = 1

    return array


# TODO: element in return matrix: index in array; array is from zig_id
def zig_id_in_m(array, shape):
    matrix = np.zeros(shape)
    for pid in range(len(array)):
        [x, y] = array[pid]
        matrix[x][y] = pid

    return matrix


# TODO: Compressed zig-F array to F matrix
def array_to_matrix(f_zig, id_matrix):
    [x, y] = id_matrix.shape
    matrix = np.zeros([x, y])
    for i in range(x):
        for j in range(y):
            matrix[i][j] = f_zig[int(id_matrix[i][j])]

    return matrix


# TODO: Discard hi frequency bands
def lossy_2d(matrix, table, zig_array, id_m, remain_rate=0.5):
    # TODO: Quantize
    table = np.array(table).astype(int)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            matrix[i][j] /= table[i][j]
    matrix = np.array(matrix).astype(int)

    one_d = []
    for [x, y] in zig_array:
        one_d.append(matrix[x][y])
    print(one_d)

    # TODO: set tail to 0 -- compress
    size = len(one_d)
    remain_len = int(size * remain_rate)
    array = np.zeros(size)
    array[0: remain_len] = one_d[0: remain_len]
    dct_compressed = array_to_matrix(array, id_m)

    # TODO: restore table
    for i in range(dct_compressed.shape[0]):
        for j in range(dct_compressed.shape[1]):
            dct_compressed[i][j] *= table[i][j]

    return dct_compressed


# TODO: F = AfA'; f = A'FA; return A
def dct_matrix(shape):
    N = shape[0]
    assert shape[1] == N
    A = np.zeros([N, N])
    for i in range(0, N):
        for j in range(0, N):
            if i == 0:
                c = np.sqrt(1/N)
            else:
                c = np.sqrt(2/N)
            A[i][j] = c * np.cos((j + 0.5)*i*np.pi/N)
    A = np.mat(A)
    return A


# TODO: adjust per block
def dct_2d():
    lena = np.array(cv2.imread(r'./source/Lena.jpg', cv2.IMREAD_GRAYSCALE), dtype='int32')
    # TODO: shift to symmetry 0
    lena -= 128
    shape = lena.shape
    assert shape[0] == shape[1] and shape[0] % block_size == 0

    dct = np.zeros(shape)
    A = dct_matrix([block_size, block_size])
    A_t = np.transpose(A)
    # TODO: transform each block
    i = 0
    while i < shape[0]:
        j = 0
        while j < shape[1]:
            f = lena[i:i+block_size, j:j+block_size]
            dct[i:i+block_size, j:j+block_size] = np.dot((np.dot(A, f)), A_t)
            j += block_size
        i += block_size

    cv2.imwrite(r'./target/lena_2d_ori.jpg', (np.clip(dct, 0., 255)).astype(np.uint8))
    return dct


# TODO: computing logic is the same as dct
def idct_2d(dct):
    shape = dct.shape
    lena_2d = np.zeros(shape)

    A = dct_matrix([block_size, block_size])
    A_t = np.transpose(A)
    i = 0
    while i < shape[0]:
        j = 0
        while j < shape[1]:
            f = dct[i:i + block_size, j:j + block_size]
            lena_2d[i:i + block_size, j:j + block_size] = np.dot((np.dot(A_t, f)), A)
            j += block_size
        i += block_size

    lena_2d += 128
    return lena_2d


if __name__ == "__main__":
    shape = [block_size, block_size]
    zig_id_array = zig_1d(shape=shape)
    id_matrix = zig_id_in_m(zig_id_array, shape)
    [x, y] = (np.array(cv2.imread(r'./source/Lena.jpg', cv2.IMREAD_GRAYSCALE), dtype='int32')).shape
    default_table = np.ones([block_size, block_size])

    '''
    # TODO: compress dct
    alpha = 0.2
    while alpha < 2:
        dct_origin = dct_2d()
        dct = np.zeros([x, y])
        i = 0
        while i < x:
            j = 0
            while j < y:
                f = dct_origin[i:i + block_size, j:j + block_size]
                dct[i:i + block_size, j:j + block_size] = lossy_2d(f, alpha * Table_8, zig_id_array, id_matrix, rate)
                j += block_size
            i += block_size
        cv2.imwrite(r'./target/lena_2d_alpha_' + str(alpha) + '_compressed_' + str(rate) + '.jpg', (np.clip(dct, 0., 255)).astype(np.uint8))

        # TODO: use compressed one to do idct
        target = idct_2d(dct)
        cv2.imwrite(r'./target/lena_2d_idct_alpha_' + str(alpha) + '_compressed_' + str(rate) + '.jpg', target)
        alpha += 0.2
    '''

    # TODO: compress dct
    dct_origin = dct_2d()
    dct = np.zeros([x, y])
    i = 0
    while i < x:
        j = 0
        while j < y:
            f = dct_origin[i:i + block_size, j:j + block_size]
            dct[i:i + block_size, j:j + block_size] = lossy_2d(f, Canon, zig_id_array, id_matrix, rate)
            j += block_size
        i += block_size
    cv2.imwrite(r'./target/lena_2d_Canon_compressed_' + str(rate) + '.jpg',
                (np.clip(dct, 0., 255)).astype(np.uint8))

    # TODO: use compressed one to do idct
    target = idct_2d(dct)
    cv2.imwrite(r'./target/lena_2d_idct_Canon_compressed_' + str(rate) + '.jpg', target)


import cv2
import numpy as np
rate = 1/1


# TODO: dct_id transform matrix: F = C_uv * row/column; row/column = A'F
def dct_matrix_1d(shape):
    [x, N] = shape
    assert x == N
    c_uv = np.zeros(shape)
    for i in range(N):
        if i == 0:
            c = np.sqrt(1/N)
        else:
            c = np.sqrt(2/N)
        for j in range(N):
            c_uv[i][j] = c * np.cos((j+0.5) * np.pi/N * i)

    return np.mat(c_uv)


def dct_idct_one_time(src, A, remain_rate=0.5):
    # TODO: dct as row (column input is transposed)
    [shape0, shape1] = src.shape
    remain_len = int(shape0 * remain_rate)
    dct = np.zeros([shape0, shape1])
    for i in range(shape0):
        row = src[i, :]
        F = np.squeeze(np.array(np.dot(A, row)))
        # TODO: compress
        dct[i][0: remain_len] = F[0: remain_len]

    dct = dct.astype(int)

    # TODO: idct for row
    A_t = np.transpose(A)
    idct = np.zeros([shape0, shape1])
    for i in range(shape0):
        row = dct[i, :]
        f = np.dot(A_t, row)
        idct[i][:] = f[:]
    return dct, idct


def dct_1d(src, A, remain_rate=0.5, direction='row'):
    if direction == 'row':
        dct_rows, idct_src = dct_idct_one_time(src, A, remain_rate)
        print("*************ROW FINISHED***************")

        dct_columns, idct_final = dct_idct_one_time(np.transpose(idct_src), A, remain_rate)
        dct_columns = np.transpose(dct_columns)
        idct_final = np.transpose(idct_final)
    else:
        dct_rows, idct_src = dct_idct_one_time(np.transpose(src), A, remain_rate)
        print("*************COLUMN FINISHED***************")
        idct_src = np.transpose(idct_src)
        dct_rows = np.transpose(dct_rows)

        dct_columns, idct_final = dct_idct_one_time(idct_src, A, remain_rate)

    return dct_rows, dct_columns, idct_src, idct_final


if __name__ == "__main__":
    source = np.array(cv2.imread(r'./source/Lena.jpg', cv2.IMREAD_GRAYSCALE), dtype='int32')
    matrix = dct_matrix_1d(source.shape)
    source -= 128
    rows, columns, middle, final = dct_1d(source, matrix, rate, 'column')

    cv2.imwrite(r'./target/lena_1d_dct_row_2_' + str(rate) + '.jpg', (np.clip(rows, 0.0, 255.0)).astype(np.int8))
    cv2.imwrite(r'./target/lena_1d_dct_column_1_' + str(rate) + '.jpg', (np.clip(columns, 0.0, 255.0)).astype(np.int8))
    middle += 128
    cv2.imwrite(r'./target/lena_1d__row_2_compressed_' + str(rate) + '.jpg', middle)
    final += 128
    cv2.imwrite(r'./target/lena_1d__final_cr_compressed_' + str(rate) + '.jpg', final)







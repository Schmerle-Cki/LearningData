import cv2
import numpy as np
import json

base_dir = r'./image fusion/'


def is_black(point, img):
    is_b = any(img[point[0]][point[1]][:] < np.array([50, 50, 50]))
    if is_b:
        return 1
    else:
        return 0


def detect_edge(img, limit):
    edge_points = []
    inner_points = []
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            i_am_back = is_black([i, j], img)
            if i_am_back:
                continue

            if limit[0][0] < i < limit[1][0] and limit[0][1] < j < limit[1][1]:
                inner_points.append([i, j])
                continue
            left_black = is_black([i-1, j], img)
            right_black = is_black([i+1, j], img)
            up_black = is_black([i, j-1], img)
            down_black = is_black([i, j+1], img)
            total = left_black + right_black + up_black + down_black
            if total == 0:
                inner_points.append([i, j])
            else:
                edge_points.append([i, j])

    return edge_points, inner_points


def dot(mask_path, source_path, post):
    mask_p = base_dir + mask_path + post
    source_p = base_dir + source_path + post

    mask_im = cv2.imread(mask_p)
    label_im = cv2.imread(source_p)

    mask = np.array(mask_im, dtype='int32')
    source = np.array(label_im, dtype='int32')

    source[:][:][:] = (source[:][:][:] * mask[:][:][:])/255
    save_path = base_dir + source_path + '_ok' + post

    print('dotted mask', save_path)
    cv2.imwrite(save_path, source)


def brutal_in(src, target, post, limit,start_x=0, start_y=0):
    pid = src[4]
    src_p = base_dir + src + post
    trg_p = base_dir + target + post
    print(src_p)
    print(trg_p)

    trg = np.array(cv2.imread(trg_p))
    src = np.array(cv2.imread(src_p))

    mask_p = r'./image fusion/test' + pid + '_mask_1' + post
    print(mask_p)
    mask = cv2.imread(mask_p)
    edge_points, inner_points = detect_edge(mask, limit)
    json.dump({'edge': edge_points, 'inner': inner_points}, open(r'./image fusion/' + target[0: 5] + '_points.json', 'w'))

    label = trg
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            point = [i, j]
            if point in edge_points or point in inner_points:
                label[i + start_x][j + start_y][:] = src[i][j][:]

    save_p = base_dir + target + '_brutal' + post
    cv2.imwrite(save_p, label)


if __name__ == "__main__":
    dot('test1_mask', 'test1_src', '.jpg')
    # TODO: adjust manually
    src_1_p = r'./image fusion/test1_src_ok.jpg'
    src_1 = np.array(cv2.imread(src_1_p), dtype='int32')
    msk_1_p = r'./image fusion/test1_mask.jpg'
    msk_1 = np.array(cv2.imread(msk_1_p), dtype='int32')

    w = src_1.shape[0]
    h = src_1.shape[1]
    print(src_1.shape)
    print(w/4, h*0.4)
    src_1 = src_1[int(w/3):][:][:]
    msk_1 = msk_1[int(w/3):][:][:]

    src_2_h = int(h*0.4)
    src_2 = np.random.randn(src_1.shape[0], src_1.shape[1] - src_2_h, 3)
    msk_2 = np.random.randn(src_1.shape[0], src_1.shape[1] - src_2_h, 3)
    for i in range(src_1.shape[0]):
        src_2[i][:][:] = src_1[i][src_2_h:][:]
        msk_2[i][:][:] = msk_1[i][src_2_h:][:]
    print(src_2.shape)
    cv2.imwrite(src_1_p, src_2)
    cv2.imwrite(r'./image fusion/test1_mask_1.jpg', msk_2)
    brutal_in('test1_src_ok', 'test1_target', '.jpg', [[0, 0], [0, 0]], 0, 0)

    dot('test2_mask', 'test2_src', '.png')
    brutal_in('test2_src_ok', 'test2_target', '.png', [[35, 75], [70, 120]], 165, 140)
    print('finished')




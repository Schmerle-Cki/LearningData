import cv2
import numpy as np
import os
import json
blk_sz = 8
epsilon = 1e-15


def grade(src_p, trg_p):
    # TODO: MSE
    src = np.array(cv2.imread(src_p, cv2.IMREAD_GRAYSCALE), dtype='int32')
    trg = np.array(cv2.imread(trg_p, cv2.IMREAD_GRAYSCALE), dtype='int32')
    mse = 0.0
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            mse += (src[i][j] - trg[i][j])**2
    mse = mse/src.shape[0]/src.shape[1]

    # TODO: PSNR
    psnr = 10*np.log10(255**2/mse)
    return mse, psnr


def blk_avg(src, trg):
    total_mse = 0.0
    total_psnr = 0.0
    [x, y] = src.shape
    blk_num = (x/blk_sz)*(y/blk_sz)

    i = 0
    while i < x:
        j = 0
        mse = 0.0
        while j < y:
            for ii in range(i, i+blk_sz):
                for jj in range(j, j+blk_sz):
                    mse += (src[ii][jj] - trg[ii][jj])**2
            mse = mse/(blk_sz**2)

            total_mse += mse
            if mse == 0:
                mse = epsilon
            total_psnr += 10*np.log10((255**2)/mse)

            j += blk_sz
        i += blk_sz

    total_mse = total_mse/blk_num
    total_psnr = total_psnr/blk_num
    return total_psnr, total_mse


if __name__ == "__main__":
    PSNR = {}

    base_dir = r'./target'
    source = np.array(cv2.imread(r'./source/Lena.jpg', cv2.IMREAD_GRAYSCALE), dtype='int32')

    for name in os.listdir(base_dir):
        if ('1d' in name and 'final' in name) or ('2d' in name and 'idct' in name):
            if 'blk' in name:
                b_id = name.index('blk')
                c_id = name.index('compressed')
                blk_sz = int(name[b_id + 4: c_id - 1])
            else:
                blk_sz = 8
            image_url = base_dir + '/' + name
            im = np.array(cv2.imread(image_url, cv2.IMREAD_GRAYSCALE), dtype='int32')
            psnr, mse = blk_avg(source, im)
            PSNR.update({name[: -4]: [psnr, mse]})

    json.dump(PSNR, open('psnr_res.json', 'w'), indent=1)

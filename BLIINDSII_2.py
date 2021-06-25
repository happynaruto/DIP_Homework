import numpy as np
import cv2
import os
from scipy.special import gamma
from multiprocessing import Pool
import time
import math
import gc
from matplotlib import pyplot as plt

'''
TODO
1. block overlap
2. img multi-scale
'''


def block_gamma(block_dct, gamma_sequence, r_lut, eps):
    # 剔除DC分量
    dct_coe = block_dct.flatten()[1:]
    m = np.mean(dct_coe)
    m_abs = np.mean(abs(dct_coe - m))
    # bug修改 var是dct_coe的方差, est_r中不用平方
    v = np.var(dct_coe)
    est_r = v / (m_abs + eps) ** 2
    gamma_index = np.argmin(abs(r_lut - est_r))
    gamma_value = gamma_sequence[gamma_index]
    return gamma_value


def block_energy(block_dct, subbands, eps):
    ori_dcts = block_dct * subbands
    var_band1, var_band2, var_band3 = np.var(ori_dcts, axis=(1, 2))
    r1 = abs(var_band3 - (var_band1 + var_band2) / 2) / (var_band3 + (var_band1 + var_band2) / 2 + eps)
    r2 = abs(var_band2 - var_band1) / (var_band3 + var_band1 + eps)
    return (r1 + r2) / 2


def block_orientation(block_dct, orientations, eps):
    ori_dcts = block_dct * orientations
    std_gauss = np.std(abs(ori_dcts), axis=(1, 2))
    mean_abs = np.mean(abs(ori_dcts), axis=(1, 2))
    return np.var(std_gauss / (mean_abs + eps))


def cal_precentile(data):
    data_sorted = np.sort(data)
    # 升序 取最后10%
    l = len(data_sorted)
    p10 = round(np.mean(data_sorted[- l // 10:]), 3)
    p100 = round(np.mean(data_sorted), 3)
    return p10, p100


def cal_features(img_path, pic_name, block_size, gamma_sequence, r_lut, eps,
                 subbands, orientations, k):
    start = time.time()
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
    gray_img = np.float32(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    h, w, c = img.shape
    # gray_img = cv2.resize(gray_img, (1000, int(h*1000/w)))
    # h, w = gray_img.shape
    gamma_data = []
    freq_data = []
    energy_data = []
    orientation_data = []
    # TODO 边缘被抛弃优化
    for row in range(0, h // block_size * block_size, block_size):
        for col in range(0, w // block_size * block_size, block_size):
            cur_block = gray_img[row:row + block_size, col:col + block_size]
            cur_dct = cv2.dct(cur_block)
            cur_gamma = block_gamma(cur_dct, gamma_sequence, r_lut, eps)
            gamma_data.append(cur_gamma)
            freq_data.append((gamma(1 / cur_gamma) * gamma(3 / cur_gamma) / (gamma(2 / cur_gamma) ** 2) - 1) ** 0.5)
            energy_data.append(block_energy(cur_dct, subbands, eps))
            orientation_data.append(block_orientation(cur_dct, orientations, eps))
    gamma_p10, gamma_p100 = cal_precentile(gamma_data)
    freq_p10, freq_p100 = cal_precentile(freq_data)
    eng_p10, eng_p100 = cal_precentile(energy_data)
    ori_p10, ori_p100 = cal_precentile(orientation_data)
    msg = [str(gamma_p10), str(gamma_p100), str(freq_p10), str(freq_p100),
           str(eng_p10), str(eng_p100), str(ori_p10), str(ori_p100)]
    end = time.time()
    print('ID %s Task %s -%i runs %0.2f seconds.' % (os.getpid(), pic_name, k, (end - start)))
    return msg


def gen_energy_mask(block_size):
    subbands = np.zeros((3, block_size, block_size))
    for i in range(block_size):
        for j in range(block_size):
            if (i <= block_size // 2) and (i <= math.ceil(block_size / 2) - j - 1):
                subbands[0, i, j] = 1
            elif (i >= block_size // 2) and (i >= math.floor(block_size * 3 / 2) - j - 1):
                subbands[2, i, j] = 1
    subbands[1, :, :] = np.ones((block_size, block_size)) - subbands[0, :, :] - subbands[2, :, :]
    subbands[:, 0, 0] = 0
    return subbands


def gen_orientation_mask(block_size):
    orientations = np.zeros((3, block_size, block_size))
    for i in range(block_size):
        for j in range(block_size):
            if i > math.floor(1.5 * j):
                orientations[2, i, j] = 1
            elif i < math.ceil(0.6 * j):
                orientations[0, i, j] = 1
    orientations[1, :, :] = np.ones((block_size, block_size)) - orientations[0, :, :] - orientations[2, :, :]
    orientations[:, 0, 0] = 0
    return orientations


if __name__ == '__main__':
    worker = 12
    pool = Pool(worker, maxtasksperchild=5)
    block_size = 5
    gamma_sequence = np.linspace(0.03, 20, num=2000)
    r_lut = gamma(1 / gamma_sequence) * gamma(3 / gamma_sequence) / (gamma(2 / gamma_sequence) ** 2)
    eps = 0.00000001
    subbands = gen_energy_mask(block_size)
    orientations = gen_orientation_mask(block_size)
    res_list = []
    t_start = time.time()
    i = 0

    # 图片列表获取 - 调用gen_pic_corresponding生成对应列表
    path = 'E:/02 Face Illumination/04 Reality/Imgs/'
    folders = os.listdir(path)
    picture_list = []
    with open('pic_corresponding_list', 'r') as img_list_file:
        while True:
            line = img_list_file.readline()[:-1].split(',')
            if len(line[0]) == 0:
                break
            picture_list.append(line)

    for picture in picture_list:
        img_name = picture[0]
        img_paths = picture[1:]
        msg = [img_name]
        k = 0  # 0 - org // 1 - ps // 2 - camera // 3 - samsung note10
        for img_path in img_paths:
            if img_path != 'Missing':
                try:
                    # img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
                    res = pool.apply_async(cal_features, (img_path, img_name, block_size, gamma_sequence, r_lut,
                                                          eps, subbands, orientations, k, ))
                    res_list.append([img_name, str(k), res])
                    i += 1
                except Exception as e:
                    res_list.append([img_name, str(k), "Na"])
                    print(img_name, k, 'Error')
                    print("exception is:" + str(e))
            else:
                res_list.append([img_name, str(k), "Na"])
            k += 1

    # 写结果
    with open('gamma_result.txt', 'w+') as f:
        for res in res_list:
            if res[2] == "Na":
                f.writelines(','.join(res[0:2]))
                f.writelines(',')
                f.writelines(','.join(['Na'] * 8) + '\n')
            else:
                f.writelines(','.join(res[0:2]))
                f.writelines(',')
                f.writelines(','.join(res[2].get()) + '\n')
    pool.close()
    pool.join()
    t_end = time.time()
    print('total time = ', t_end - t_start)
    print('avg time = ', (t_end - t_start) / i)

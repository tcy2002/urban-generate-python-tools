import math

import cv2
import numpy as np


# estimate maximum number of seeds
def _estimate_max_seeds(image, thickness=1):
    total = np.sum(image == 255)
    x, y = image.shape[:2]
    return int(total / math.log(x * y) / 2 / thickness)


# generate random seeds
def _generate_seeds(image, num_seeds):
    seeds = []
    for i in range(num_seeds):
        x = np.random.randint(0, image.shape[1])
        y = np.random.randint(0, image.shape[0])
        seeds.append((x, y))
    return seeds


# iter 1: dropout seeds near roads
def _dropout_seeds1(image, seeds, radius):
    mask = cv2.morphologyEx(image, cv2.MORPH_DILATE, np.ones((radius, radius), np.uint8))
    new_seeds = []
    for seed in seeds:
        if mask[seed[1], seed[0]] == 0:
            new_seeds.append(seed)
    return new_seeds


# iter 2: dropout seeds far from roads
def _dropout_seeds2(image, seeds, radius):
    mask = cv2.morphologyEx(image, cv2.MORPH_DILATE, np.ones((radius, radius), np.uint8))
    new_seeds = []
    for seed in seeds:
        if mask[seed[1], seed[0]] == 255:
            new_seeds.append(seed)
    return new_seeds


# iter 3: dropout seeds that are too close to each other
def _dropout_seeds3(seeds, radius):
    new_seeds = []
    for seed in seeds:
        for other_seed in seeds:
            if seed != other_seed and np.linalg.norm(np.array(seed) - np.array(other_seed)) < radius:
                break
        else:
            new_seeds.append(seed)
    return new_seeds


# draw seeds on image
def _draw_seeds(image, seeds):
    for seed in seeds:
        cv2.circle(image, seed, 1, 255, -1)
    return image


# main
def generate_seeds(image_name, radius1, radius2, radius3, thickness=1):
    image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    max_n = _estimate_max_seeds(image, thickness)
    print('最大种子数目', max_n)

    seeds = []
    num = 0
    while len(seeds) < max_n:
        seeds.extend(_generate_seeds(image, 50))
        seeds = _dropout_seeds1(image, seeds, radius1)
        seeds = _dropout_seeds2(image, seeds, radius2)
        seeds = _dropout_seeds3(seeds, radius3)

        # 最多迭代50次
        num += 1
        if num > 50:
            break

    print('实际种子数目', len(seeds))
    _draw_seeds(image, seeds)
    return seeds, image

from scipy.spatial import Delaunay
import numpy as np
import matplotlib.pyplot as plt
import json


def to_mesh(points):
    points = np.array(points)
    tri = Delaunay(points)

    plt.triplot(points[:, 0], points[:, 1], tri.simplices.copy())
    plt.plot(points[:, 0], points[:, 1], 'o')
    plt.show()

    return tri


def from_img_point_to_mesh(source, target, interpolation):
    middle = (1-interpolation) * source + interpolation * target
    tri = to_mesh(middle)
    return tri


def load_tri_array(source_name, base_dir=r'./face morphing/'):
    file_name = base_dir + source_name + '_array.json'
    source = dict(json.load(open(file_name, 'r')))['data']
    return source

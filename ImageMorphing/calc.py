import numpy as np
import scipy.linalg as linalg


# 参数分别是旋转轴和旋转弧度值
def rotate_mat(axis, radian):
    # rot_matrix = linalg.expm(np.cross(np.eye(3), axis / linalg.norm(axis) * radian))
    cos = np.cos(radian)
    sin = np.sin(radian)
    t = 1 - cos
    x = axis[0]
    y = axis[1]
    rot_matrix = np.array([[t*x*x+cos, t*x*y, sin*y],
                           [t*x*y, t*y*y+cos, -sin*x],
                           [-sin*y, sin*x, cos]])
    return rot_matrix


# 分别是x,y和z轴,也可以自定义旋转轴
'''axis_x, axis_y, axis_z = [1,0,0], [0,1,0], [0, 0, 1]
# pi/4
yaw = 0.7854
# 绕Z轴旋转pi/4
matrix = rotate_mat([0, 1, 1], yaw*2)
print(matrix)
x = np.array((1, 0, 0))
print(np.dot(matrix, x))'''


import dlib
import cv2
import json
from PIL import Image

predictor_p = r'./model/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_p)


# TODO: resize the image
def resize_to_same(src, trg):
    img_s = Image.open(src)
    img_t = Image.open(trg)
    w = min(img_s.width, img_t.width)
    h = min(img_s.height, img_t.height)
    try:
        new_img = img_s.resize((w, h), Image.BILINEAR)
        new_img.save(src)

        new_img = img_t.resize((w, h), Image.BILINEAR)
        new_img.save(trg)
    except Exception as e:
        print(e)


# TODO: Extract Feature points in each raw picture and save the labeled img + feature points json file
# TODO: 'x' is horizontal
def extract_one(file_path, add_pts, predict=True):
    file = base_dir + file_path
    img = cv2.imread(file)

    if predict:
        faces = detector(img, 2)
        window = dlib.image_window()

        window.clear_overlay()
        window.set_image(img)

        for index, face in enumerate(faces):
            # Get key points
            shape = predictor(img, face)
        try:
            points = [[pt.x, pt.y] for pt in shape.parts()]
        except:
            points = []
    else:
        points = []

    shape = img.shape
    half_x = int(shape[0]/2)
    half_y = int(shape[1]/2)
    # TODO: the order of cv2 and points differs
    points.extend(add_pts)
    points.extend([[0, 0], [shape[1] - 1, 0], [shape[1] - 1, shape[0] - 1], [0, shape[0] - 1], [0, half_x],
                   [shape[1] - 1, half_x], [half_y, 0], [half_y, shape[0] - 1]])

    for point in points:
        cv2.circle(img, (point[0], point[1]), 2, (0, 255, 0), 1)

    cv2.imwrite(base_dir + file_path[0: -4] + '_labeled.png', img)

    json.dump({'data': points}, open(base_dir + file_path + '_array.json', 'w'))


# TODO: Load json {x:[], y:[]} and pair them to points
# TODO: 'x' is horizontal
def from_dict_to_points(points_dict):
    points_dict = dict(points_dict)
    x = points_dict['x']
    y = points_dict['y']
    points = []
    for i in range(len(x)):
        points.append([int(x[i]), int(y[i])])
    return points


if __name__ == '__main__':
    base_dir = r'./face morphing/'
    resize_to_same(base_dir + 'source1.png', base_dir + 'target1.png')
    resize_to_same(base_dir + 'source2.png', base_dir + 'target2.png')

    points_src_1_dict = dict(json.load(open(r'./face morphing/source1_hand_points.json', 'r')))
    points_trg_1_dict = dict(json.load(open(r'./face morphing/target1_hand_points.json', 'r')))
    points_trg_2_dict = dict(json.load(open(r'./face morphing/target2_hand_points.json', 'r')))
    src_1_points = from_dict_to_points(points_src_1_dict)
    trg_1_points = from_dict_to_points(points_trg_1_dict)
    trg_2_points = from_dict_to_points(points_trg_2_dict)

    # extract_one('source1.png', [])
    # extract_one('target1.png', [])
    # extract_one('source2.png', [])
    # extract_one('target2.png', trg_2_points)

    base_dir = r'./view morphing/'
    resize_to_same(base_dir + 'source_1.png', base_dir + 'target_1.png')
    resize_to_same(base_dir + 'source_2.png', base_dir + 'target_2.png')
    points_src_1_dict = dict(json.load(open(r'./view morphing/source_1_hand_points.json', 'r')))
    points_src_2_dict = dict(json.load(open(r'./view morphing/source_2_hand_points.json', 'r')))
    points_trg_1_dict = dict(json.load(open(r'./view morphing/target_1_hand_points.json', 'r')))
    points_trg_2_dict = dict(json.load(open(r'./view morphing/target_2_hand_points.json', 'r')))
    src_1_points = from_dict_to_points(points_src_1_dict)
    src_2_points = from_dict_to_points(points_src_2_dict)
    trg_1_points = from_dict_to_points(points_trg_1_dict)
    trg_2_points = from_dict_to_points(points_trg_2_dict)

    extract_one('source_1.png', src_1_points, False)
    extract_one('target_1.png', trg_1_points, False)
    extract_one('source_2.png', src_2_points)
    extract_one('target_2.png', trg_2_points, False)




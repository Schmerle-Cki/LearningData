import cv2
import json


# TODO: when labeling, the points order in source and target must be the same
# TODO: when labeling, 'x' is shorter than 'y', which means 'x' is horizontal, differs with cv2.read!
def label_one_img(img_name):
    img = cv2.imread(img_name)
    a = []
    b = []

    def on_event_l_button_down(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            xy = "%d,%d" % (x, y)
            a.append(x)
            b.append(y)
            cv2.circle(img, (x, y), 2, (0, 255, 0), 1)
            cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 0), thickness=1)
            cv2.imshow("image", img)

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_event_l_button_down)
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    for i in range(len(a)):
        print(a[i], b[i])

    json.dump({'x': a, 'y': b}, open(img_name[:-4] + '_hand_points.json', 'w'))
    return a, b


if __name__ == '__main__':
    # label_one_img(r'./face morphing/target2.png')
    # label_one_img(r'./view morphing/source_1.png')
    # label_one_img(r'./view morphing/target_1.png')
    label_one_img(r'./view morphing/source_2.png')
    # label_one_img(r'./view morphing/target_2.png')


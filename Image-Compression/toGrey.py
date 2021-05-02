import cv2

if __name__ == "__main__":
    img_path = r'../ClassicalImage/Lena.jpg'
    lena = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    cv2.imshow("lena_grey", lena)
    cv2.waitKey()
    cv2.imwrite(r'./source/Lena.jpg', lena)

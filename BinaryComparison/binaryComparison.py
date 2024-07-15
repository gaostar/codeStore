import cv2
import numpy as np

def check_image_sizes(img1, img2):
    if img1.shape[:2] != img2.shape[:2]:
        return False
    return True

def create_comparison_image(ref_binary, test_binary, color, alpha=0.8):
    if color:
        comparison = np.zeros((ref_binary.shape[0], ref_binary.shape[1], 3), dtype=np.uint8)
        comparison[..., 0] = ref_binary   # 蓝色
        comparison[..., 2] = test_binary   # 红色
    else:
        comparison = cv2.addWeighted(ref_binary, alpha, test_binary, 1-alpha, 0)
    return comparison


if __name__ == "__main__":

    image_ref = cv2.imread("1.jpg")
    image_test = cv2.imread("2.jpg")

    # 检查图像尺寸
    if not check_image_sizes(image_ref, image_test):
        print("Inconsistent image size, adjusting test image size.......")
        image_test = cv2.resize(image_test, (image_ref.shape[1], image_ref.shape[0]))

    ref_img_gray = cv2.cvtColor(image_ref, cv2.COLOR_BGR2GRAY)
    test_img_gray = cv2.cvtColor(image_test, cv2.COLOR_BGR2GRAY)
    binary_threshold, ref_img_binary = cv2.threshold(ref_img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    test_img_binary = cv2.threshold(test_img_gray, binary_threshold, 255, cv2.THRESH_BINARY)[1]

    # 创建比较图像
    color_flag = True   # 设置为 True 以使用彩色显示，False为白色显示
    comparison_img = create_comparison_image(ref_img_binary, test_img_binary, color_flag)

    # 显示
    cv2.namedWindow("Comparison Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Comparison Image", comparison_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 保存
    # cv2.imwrite("comparison_img.png" ,comparison_img )

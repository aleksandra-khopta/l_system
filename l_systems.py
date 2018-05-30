import cv2
import numpy as np
import math
import image_similarity
# from PIL import Image, ImageDraw

image_src = cv2.imread("fractal.jpg")
visited = np.zeros(image_src.shape[:2])
h, w = image_src.shape[:2]

def subimage(image, center, theta, width, height):
    if 45 < theta <= 90:
        theta = theta - 90
        width, height = height, width

    theta *= math.pi / 180 # convert to rad
    v_x = (math.cos(theta), math.sin(theta))
    v_y = (-math.sin(theta), math.cos(theta))
    s_x = center[0] - v_x[0] * (width / 2) - v_y[0] * (height / 2)
    s_y = center[1] - v_x[1] * (width / 2) - v_y[1] * (height / 2)
    mapping = np.array([[v_x[0],v_y[0], s_x], [v_x[1],v_y[1], s_y]])
    return cv2.warpAffine(image, mapping, (width, height), flags=cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)

################### TO DO #################
def subimage_best(image, bottom_center, theta, scale):
    width = int(w * scale)
    height = int(h * scale)

    if 45 < theta <= 90:
        theta = theta - 90
        width, height = height, width
    theta *= math.pi / 180 # convert to rad

    bottom_x, bottom_y = bottom_center
    center_x = bottom_x + (height // 2) * math.sin(theta)
    center_y = bottom_y - (height // 2) * math.cos(theta)
    center = (int(center_x), int(center_y))


    dbg_image = np.copy(image)
    cv2.circle(dbg_image, center, 3, (0, 0, 255), thickness=3)
    cv2.circle(dbg_image, bottom_center, 3, (0, 200, 200), thickness=3)
#    cv2.imshow("Debug", dbg_image)
#    cv2.waitKey()


    v_x = (math.cos(theta), math.sin(theta))
    v_y = (-math.sin(theta), math.cos(theta))
    s_x = center[0] - v_x[0] * (width / 2) - v_y[0] * (height / 2)
    s_y = center[1] - v_x[1] * (width / 2) - v_y[1] * (height / 2)
    mapping = np.array([[v_x[0],v_y[0], s_x], [v_x[1],v_y[1], s_y]])
    if image is None:
        print("Error")
    else:
        print("shape for warp = {}".format(str(image.shape)))

    return cv2.warpAffine(image, mapping, (width, height), flags=cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)

subimage1 = subimage(image_src, (310, 270), 25, w // 2, h // 2)
subimage2 = subimage(image_src, (200, 270), -25, w // 2, h // 2)
subimage3 = subimage(image_src, (255, 130), 0, w // 2, h // 2)
origin45 = subimage(image_src, (w // 2, h // 2), 45, w, h)
subimage4 = subimage_best(image_src, (255, 385), 25, 0.5)
cv2.imwrite("subimage1.jpg", subimage1)
cv2.imwrite("subimage2.jpg", subimage2)
cv2.imwrite("subimage3.jpg", subimage3)
cv2.imwrite("subimage4.jpg", subimage4)
cv2.imwrite("origin45.jpg", origin45)

subimages = [[310, 270, 25, w // 2, h //2], [200, 270, -25, w // 2, h //2], [255, 130, 0, w // 2, h //2]]

#Clock-wise from the bottom center

subimages_num = 3
subimages_bb = {}
#get bounding boxes
#get start bb index
# lowest_bb_index = 0;
# min_bb_y = 100000;
# for i in range(1, subimages_num)  :
#     width = subimages[i][3]
#     height = subimages[i][4]
#     bb_left = subimages[i][1] - width // 2
#     bb_right = subimages[i][1] + width // 2
#     bb_bottom = subimages[i][1] + height // 2
#     bb_top = subimages[i][1] - height // 2
#     if (bb_bottom < min_bb_y):
#         min_bb_y = bb_bottom
#         lowest_bb_index = i
#     subimages_bb.add({bb_left, bb_top, bb_right, bb_bottom})

#if not symmetric - start is always from bottom

# l_system1 = {""}
# l_system2 = {""}
# for i in range(1, subimages_num)  :
#     center = (subimages[i][0],subimages[i][1])
#     theta = subimages[i][2]
#     width = subimages[i][3]
#     height = subimages[i][4]
#     if (theta > 0):
#         l_system1.add("+")
#     elif (theta < 0):
#         l_system1.add("-")
#     l_system1.add("X")
#     n = w // width
#     m = h // height
#     if (m == n):
#         for j in range(1, n):
#             l_system2.add("F")

#####################################################
# image = Image.open("image.jpg") #Открываем изображение.
# draw = ImageDraw.Draw(image) #Создаем инструмент для рисования.
# width = image.size[0] #Определяем ширину.
# height = image.size[1] #Определяем высоту.
# pix = image.load() #Выгружаем значения пикселей.
# image.save("res.jpg", "JPEG")

class color(object):
    def __init__(self, r, g, b):
        self.r = r
        self.g = g
        self.b = b

def equal(image, subimage):
    similarity, corr = image_similarity.is_sample_in_image(image, subimage)
    return similarity > 0.8

def find_elem(x_start, y_start):
    bottom_center = (x_start, y_start)
    for theta in range(-90, 90):
        for scale in np.arange(0.1, 0.9, 0.1):
            sub_img = subimage_best(image_src, bottom_center, theta, scale)
            if equal(image_src, sub_img):
                return True, theta, scale
    return False

l_system1 = []


def bypass(back_color, x, y):
    if x < 0 or y < 0 or x >= w or y >= h:
        return
    if visited[y, x] == 1:
        return
    if image_src[y, x, 0] != back_color[0]:
        visited[y, x] = 1
        is_sub_img, theta, scale = find_elem(x, y)
        if is_sub_img:
            if theta > 0:
                l_system1.append("+")
            elif theta < 0:
                l_system1.append("-")
            l_system1.append("X")
            ######## TO DO : search in the frame ##########
            bypass(back_color, x + 1, y)
            bypass(back_color, x - 1, y)
            bypass(back_color, x, y + 1)
            bypass(back_color, x, y - 1)

print(image_src.shape)
print(visited.shape)

bypass((255, 255, 255), w // 2, h - 1)
print(l_system1)

import numpy as np
import cv2
from math import cos, sin, radians
import matplotlib.pyplot as plt

#读取目标图片
target = cv2.imread("image_data/111.png")
target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
target_gray = cv2.adaptiveThreshold(target_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 17, 2)

fig1 = plt.figure()
plt.axis('equal')
plt.imshow(target_gray, cmap='gray')
plt.show()

x_ = y_ = 12

part_1 = np.zeros((x_, y_))
part_2 = np.ones((x_, y_)) * 255
template_up = np.hstack((part_1, part_2))
template_low = np.hstack((part_2, part_1))

template_ = np.vstack((template_up, template_low))
template_new = template_.astype('uint8')

fig2 = plt.figure()
plt.axis('equal')
plt.imshow(template_, cmap='gray')
plt.show()

cx = template_.shape[1]
cy = template_.shape[0]

min_val_initial = 10**10000

rotation_matrix_ = cv2.getRotationMatrix2D((int(cx / 2), int(cy / 2)), 0, 2)
rotated_ = cv2.warpAffine(template_new, rotation_matrix_, (cx, cy))
ret_rotation, rotated_ = cv2.threshold(rotated_, 150, 255, cv2.THRESH_BINARY)

result = cv2.matchTemplate(target_gray, rotated_, cv2.TM_SQDIFF)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

min_val_initial = min_val
left_corner = min_loc
rotated_draw = rotated_
center_loc = np.array(list(min_loc)) + np.array([cx / 2, cy / 2])
circle_center = (int(min_loc[0]+cx/2), int(min_loc[1]+cy/2))
up_corner = (min_loc[0] + cx, min_loc[1] + cy)

strmin_val = str(min_val_initial)

cv2.rectangle(target, left_corner, up_corner, (0, 0, 225), 2)
cv2.circle(target, circle_center, 1, (0, 0, 225), -1)
cv2.imwrite('test1.jpg', target)
print(circle_center, "角点")
cv2.imshow("MatchResult----MatchingValue="+strmin_val, target)
cv2.waitKey()
cv2.destroyAllWindows()
from Helping import *
import cv2

F16_NOISY = 'res/f16noisy.gif'
MEDIAN_OUTPUT = 'out/median_filter.png'
MAX_OUTPUT = 'out/max_filter.png'
MIN_OUTPUT = 'out/min_filter.png'

f16_noisy = read_GIF(F16_NOISY)

f16_median = cv2.medianBlur(f16_noisy.astype('uint8'), 3)
kernel = np.ones((3, 3), np.uint8)
f16_max = cv2.dilate(f16_noisy.astype('uint8'), kernel)
f16_min = cv2.erode(f16_noisy.astype('uint8'), kernel)

cv2.imwrite(MEDIAN_OUTPUT, f16_median)
cv2.imwrite(MAX_OUTPUT, f16_max)
cv2.imwrite(MIN_OUTPUT, f16_min)

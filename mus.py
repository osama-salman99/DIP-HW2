import numpy as np
from Helping import *

ORIGINAL_IMAGE_PATH = "res/f16.gif"
# img = cv2.imread("/Users/user/Desktop/photo2.jpg",0)
img = read_GIF(ORIGINAL_IMAGE_PATH)
DFT = np.fft.fft2(img)
Image_magnitude = np.abs(DFT)
loged = np.log(Image_magnitude - np.min(Image_magnitude) + 1)
scaled = ((255 / (np.max(loged) - np.min(loged))) * (loged - np.min(loged)))
fshift = np.fft.fftshift(scaled)

cv2.imshow("image", fshift.astype('uint8'))
cv2.waitKey(0)
cv2.destroyAllWindows()

#################################################### PROBLEM TWO C

# total_Power = np.sum(Image_magnitude**2)
# print(total_Power.round(2))

#################################################### PROBLEM TWO D

imaginary_part = np.arctan2(DFT.imag, DFT.real)

ii = np.real(np.fft.ifft2(np.exp(1j * imaginary_part)))

# iiii = np.real(np.fft.ifft2(np.exp(1j*imaginary_part)))
scaled = ((255 / (np.max(ii) - np.min(ii))) * (ii - np.min(ii)))

cv2.imshow("image", scaled.astype('uint8'))
cv2.waitKey(0)
cv2.destroyAllWindows()

#################################################### PROBLEM TWO E
Reconstructed_phase = np.real(np.fft.ifft2(np.multiply(Image_magnitude, np.exp(0.1 * 1j * imaginary_part))))

cv2.imshow("image", Reconstructed_phase.astype('uint8'))
cv2.waitKey(0)
cv2.destroyAllWindows()

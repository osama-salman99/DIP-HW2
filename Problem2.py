from numpy import fft
from Helping import *

ORIGINAL_IMAGE_PATH = 'res/f16.gif'


def part_a():
    image = read_GIF(ORIGINAL_IMAGE_PATH)
    show_image_and_wait('Original Image', image)
    return image


def part_b():
    ft = fft.fft2(original_image)
    magnitude = get_shifted_image_magnitude(ft)

    show_image_and_wait('Image Magnitude', magnitude)

    return ft, magnitude


def part_c():
    total_Power = np.sum(image_magnitude ** 2)
    print(f'total power = {total_Power.round(2)}')


def part_d():
    phase = np.arctan2(ft_image.imag, ft_image.real)
    phase_inverse = np.real(fft.ifft2(np.exp(1j * phase)))
    scaled = linear_scale_image(phase_inverse)

    show_image_and_wait('Phase Component', scaled)

    return phase


def part_e():
    reconstructed_phase = np.real(fft.ifft2(image_magnitude * np.exp(0.1j * image_phase)))
    show_image_and_wait('Reconstructed Phase', reconstructed_phase)


def get_shifted_image_magnitude(ft):
    scaled = log_scale_image(ft)
    shifted = fft.fftshift(scaled)
    return shifted


def log_scale_image(image):
    image = abs(image)
    log_image = np.log(image - image.min() + 1)
    linear_image = linear_scale_image(log_image)
    return linear_image


def linear_scale_image(image):
    image -= image.min()
    image = c_linear(image) * image
    return image


def show_image_and_wait(label, image):
    cv2.imshow(label, image.astype('uint8'))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


original_image = part_a()
ft_image, image_magnitude = part_b()
part_c()
image_phase = part_d()
part_e()

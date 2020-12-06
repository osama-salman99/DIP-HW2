from numpy import fft
from Helping import *

ORIGINAL_IMAGE_PATH = 'res/f16.gif'


def part_a():
    image = read_GIF(ORIGINAL_IMAGE_PATH)
    show_image_and_wait('Original Image', image)
    return image


def part_b():
    ft = fft.fft2(original_image)
    magnitude = np.abs(ft)
    shifted_magnitude = get_shifted_image_magnitude(ft)

    show_image_and_wait('Image Magnitude', shifted_magnitude)

    return ft, np.abs(magnitude)


def part_c():
    total_power = np.sum(image_magnitude ** 2)
    dc_power = image_magnitude[0, 0]
    percentage = ((dc_power / total_power) * 100)

    print(f'total power = {total_power.round(2)}')
    print(f'DC power component = {dc_power}')
    print(f'The DC power component is {percentage}% of the total power.')


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

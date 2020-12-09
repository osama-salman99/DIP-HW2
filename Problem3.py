from Helping import *
from numpy import fft

CAMERAMAN_NOISY_PATH = 'res/cameramanNoisy.gif'
CAMERAMAN_ORIGINAL_PATH = 'res/cameraman.gif'


def part_a():
    lowpass_mask = ideal_lowpass_filter(magnitude, 220)
    evaluate_mask(lowpass_mask, 'ideal lowpass filter')


def part_b():
    notches_centers = ((411, 307), (307, 411), (102, 819), (205, 921), (0, 922), (102, 1023))
    notches_radii = (10, 10, 5, 5, 5, 5)
    notch_mask = ideal_notch_reject_filter(magnitude, notches_centers, notches_radii)
    evaluate_mask(notch_mask, 'ideal notch-reject filter')


def part_c():
    bands_radii = ((215, 235), (505, 525))
    band_mask = ideal_band_reject_filter(magnitude, bands_radii)
    evaluate_mask(band_mask, 'ideal band-reject filter')


def evaluate_mask(mask, filter_name):
    filtered_magnitude = log_scale_image(magnitude) * mask
    filtered_image = linear_scale_image(remove_zero_padding(get_inverse_ft(magnitude * mask, phase)))

    mask_scaled = linear_scale_image(mask)
    show_image_and_wait(mask_scaled, filter_name)
    show_image_and_wait(filtered_magnitude, f'spectrum filtered with {filter_name}')
    show_image_and_wait(filtered_image, f'image filtered with {filter_name}')

    cv2.imwrite(f'out/{filter_name}.png', mask_scaled)
    cv2.imwrite(f'out/{filter_name} spectrum.png', filtered_magnitude)
    cv2.imwrite(f'out/{filter_name} image.png', filtered_image)

    print(f'PSNR for {filter_name} = {PSNR(original_image, filtered_image)}')


def ideal_notch_reject_filter(image, centers, radii):
    middle_y, middle_x = get_middle(image)
    i, j = np.indices(image.shape, sparse=True)
    mask = np.ones(image.shape)
    for center, radius in zip(centers, radii):
        y, x = center
        y_inverse = 2 * middle_y - y
        x_inverse = 2 * middle_x - x
        distance = np.sqrt((i - y) ** 2 + (j - x) ** 2)
        distance_inverse = np.sqrt((i - y_inverse) ** 2 + (j - x_inverse) ** 2)
        mask *= np.where(distance <= radius, 0, 1)
        mask *= np.where(distance_inverse <= radius, 0, 1)
    return mask


def ideal_lowpass_filter(image: np.ndarray, radius):
    middle_y, middle_x = get_middle(image)
    lowpass_filter = 1 - ideal_notch_reject_filter(image, ((middle_y, middle_x),), (radius,))
    return lowpass_filter


def ideal_band_reject_filter(image: np.ndarray, radii):
    mask = np.ones(magnitude.shape)
    i, j = np.indices(image.shape, sparse=True)
    for inner_radius, outer_radius in radii:
        middle_y, middle_x = get_middle(image)
        distance = np.sqrt((i - middle_y) ** 2 + (j - middle_x) ** 2)
        mask *= np.where(np.logical_and(inner_radius <= distance, distance <= outer_radius), 0.0, 1.0)
    return mask


noisy_image = read_GIF(CAMERAMAN_NOISY_PATH)
original_image = read_GIF(CAMERAMAN_ORIGINAL_PATH)
padded_image = zero_pad_image(noisy_image)

ft = fft.fft2(padded_image)
magnitude, phase = get_dft_components(ft)

part_a()
part_b()
part_c()

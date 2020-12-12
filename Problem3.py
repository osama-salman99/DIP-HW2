from Helping import *

CAMERAMAN_NOISY_PATH = 'res/cameramanNoisy.gif'
CAMERAMAN_ORIGINAL_PATH = 'res/cameraman.gif'


def part_a():
    lowpass_filter_mask = ideal_lowpass_filter(magnitude, 228)
    evaluate_filter(lowpass_filter_mask, 'ideal lowpass filter')


def part_b():
    notches_centers = ((411, 307), (307, 411), (102, 819), (205, 921), (0, 922), (102, 1023))
    notches_radii = (10, 10, 5, 5, 5, 5)
    notch_filter_mask = ideal_notch_reject_filters(magnitude, notches_centers, notches_radii)
    evaluate_filter(notch_filter_mask, 'ideal notch-reject filter')


def part_c():
    bands_radii = ((215, 235), (505, 525))
    band_filter_mask = ideal_band_reject_filter(magnitude, bands_radii)
    evaluate_filter(band_filter_mask, 'ideal band-reject filter')


def evaluate_filter(filter_mask, filter_name):
    filtered_magnitude = magnitude * filter_mask
    filtered_image = linear_scale_image(remove_zero_padding(get_inverse_ft(magnitude * filter_mask, phase)))

    filter_mask_scaled = linear_scale_image(filter_mask)
    filtered_magnitude_scaled = log_scale_image(filtered_magnitude)
    show_image_and_wait(filter_mask_scaled, filter_name)
    show_image_and_wait(filtered_magnitude_scaled, f'magnitude filtered with {filter_name}')
    show_image_and_wait(filtered_image, f'image filtered with {filter_name}')

    cv2.imwrite(f'out/{filter_name}.png', filter_mask_scaled)
    cv2.imwrite(f'out/{filter_name} magnitude.png', filtered_magnitude_scaled)
    cv2.imwrite(f'out/{filter_name} image.png', filtered_image)

    print(f'PSNR for {filter_name} = {PSNR(original_image, filtered_image).round(2)}')
    print(f'{filter_name} kept {get_mask_size(filter_mask).round(2)}% of the original magnitude')


def get_mask_size(mask):
    height, width = mask.shape[:2]
    white_count = np.sum(mask)
    white_percentage = white_count / (height * width)
    return white_percentage * 100


def ideal_notch_reject_filter(image, center, radius):
    i, j = np.indices(image.shape)
    y, x = center
    distance = np.sqrt((i - y) ** 2 + (j - x) ** 2)
    filter_mask = np.where(distance <= radius, 0, 1)
    return filter_mask


def ideal_notch_reject_filters(image, centers, radii):
    middle_y, middle_x = get_middle(image)
    filter_mask = np.ones(magnitude.shape)
    for center, radius in zip(centers, radii):
        y, x = center
        y_inverse = 2 * middle_y - y
        x_inverse = 2 * middle_x - x
        filter_mask *= ideal_notch_reject_filter(image, center, radius)
        filter_mask *= ideal_notch_reject_filter(image, (y_inverse, x_inverse), radius)
    return filter_mask


def ideal_lowpass_filter(image, radius):
    middle_y, middle_x = get_middle(image)
    lowpass_filter = 1 - ideal_notch_reject_filter(image, (middle_y, middle_x), radius)
    return lowpass_filter


def ideal_band_reject_filter(image, radii):
    filter_mask = np.zeros(magnitude.shape)
    for inner_radius, outer_radius in radii:
        middle_y, middle_x = get_middle(image)
        inner_filter_mask = ideal_notch_reject_filter(image, (middle_y, middle_x), inner_radius)
        outer_filter_mask = ideal_notch_reject_filter(image, (middle_y, middle_x), outer_radius)
        band_filter_mask = inner_filter_mask - outer_filter_mask
        filter_mask += band_filter_mask
    return 1 - filter_mask


noisy_image = read_GIF(CAMERAMAN_NOISY_PATH)
original_image = read_GIF(CAMERAMAN_ORIGINAL_PATH)
padded_image = zero_pad_image(noisy_image)

ft = fft.fft2(padded_image)
magnitude, phase = get_dft_components(ft)

part_a()
part_b()
part_c()

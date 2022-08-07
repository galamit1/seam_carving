import time
from typing import Dict, Any

import numpy as np

from utils import *

NDArray = Any


def image_rotate(image: NDArray, rotate: str):
    if rotate == 'right':
        return np.rot90(image, 3)
    if rotate == 'left':
        return np.rot90(image)


def update_seam_list(seams_list: list, seam: NDArray):
    seams_list.append(seam)


def update_images(energy: NDArray, grayscale: NDArray, index_matrix: NDArray, seam: NDArray):
    updated_energy = np.zeros(shape=(energy.shape[0], energy.shape[1] - 1))
    updated_grayscale = np.zeros(shape=(grayscale.shape[0], grayscale.shape[1] - 1))
    updated_indexes = np.zeros(shape=(index_matrix.shape[0], index_matrix.shape[1] - 1), dtype=int)
    for i in range(energy.shape[0]):
        updated_energy[i] = np.hstack((energy[i, :seam[i]], energy[i, seam[i] + 1:]))
        updated_grayscale[i] = np.hstack((grayscale[i, :seam[i]], grayscale[i, seam[i] + 1:]))
        updated_indexes[i] = np.hstack((index_matrix[i, :seam[i]], index_matrix[i, seam[i] + 1:]))

    return updated_energy, updated_grayscale, updated_indexes


def optimal_seam_backtrack(path_matrix: NDArray, index_matrix: NDArray, color_matrix: NDArray, seam_matrix: NDArray, color: NDArray, j: int, seam_counter: int):
    # calculate the seam indexes according to the M indexes matrix
    seam = np.zeros(shape=(path_matrix.shape[0]), dtype=int)
    for i in range(path_matrix.shape[0] - 1, -1, -1):
        index = index_matrix[i][j]
        seam_matrix[i][seam_counter] = index
        color_matrix[i][index] = color
        seam[i] = j
        j += path_matrix[i][j]
    return seam


def get_c_matrixes(greyscale: NDArray) -> (NDArray, NDArray, NDArray):
    greyscale = np.insert(greyscale, 0, greyscale[:, 1] - 255, axis=1)
    greyscale = np.insert(greyscale, greyscale.shape[1], greyscale[:, greyscale.shape[1] - 2] + 255, axis=1)
    cv = np.abs(greyscale[:, 2:] - greyscale[:, :-2])
    greyscale = np.insert(greyscale, 0, 0, axis=0)
    cl = cv + np.abs(greyscale[:-1, 1:-1] - greyscale[1:, :-2])
    cr = cv + np.abs(greyscale[:-1, 1:-1] - greyscale[1:, 2:])
    return cl, cv, cr


def forward_looking_energy_function(energy: NDArray, grayscale: NDArray, forward_implementation: bool) -> (NDArray, NDArray):
    M = energy.copy()  # start with E matrix
    M_indexes = np.zeros_like(M)  # from which indexes we took the min values
    cl, cv, cr = get_c_matrixes(grayscale) if forward_implementation else (np.zeros_like(M), np.zeros_like(M), np.zeros_like(M))

    for i in range(1, energy.shape[0]):
        # calculate minimum solutions from i-1 + c_matrixes
        previous_row = M[i-1]
        row_with_cl = np.concatenate((np.array([np.max(previous_row) + 1]), previous_row[:-1]), axis=0) + cl[i]
        row_with_cv = previous_row + cv[i]
        row_with_cr = np.concatenate((previous_row[1:], np.array([np.max(previous_row) + 1])), axis=0) + cr[i]
        M[i] = np.minimum(np.minimum(row_with_cl, row_with_cv), row_with_cr)  # take the minimum option in every index

        # update the indexes of the min arguments
        cl_indexes = ((M[i] - row_with_cl) == 0) * -1
        cv_indexes = ((M[i] - row_with_cv) != 0) * 1
        cr_indexes = ((M[i] - row_with_cr) == 0) * 1
        M_indexes[i] = cl_indexes + cr_indexes + (-cl_indexes == cr_indexes) * cl_indexes * cv_indexes

    return M, M_indexes.astype(int)


def create_seams_image(image: NDArray, color_matrix: NDArray, axis: str):
    if axis == 'width':
        temp = color_matrix + ((color_matrix == 1) * 255)
        mark_image = np.minimum(image, temp)
        temp = color_matrix - ((color_matrix == 1) * 1)
        mark_image = np.maximum(mark_image, temp)
    else:
        temp = color_matrix + ((color_matrix == 1) * 255)
        mark_image = np.minimum(image, temp)

    return mark_image


def resize_image_by_seams(image: NDArray, seam_matrix: NDArray, k: int, remove: bool):
    resize_image = np.zeros(shape=(image.shape[0], image.shape[1] - k, 3))
    if remove:
        for i in range(image.shape[0]):
            mask = np.ones(image.shape[1], dtype=bool)
            mask[seam_matrix[i]] = False
            resize_image[i] = image[i, mask]
    else:
        for i in range(image.shape[0]):
            temp = image[i][:seam_matrix[i][0] + 1]
            temp = np.vstack((temp, image[i][seam_matrix[i][0]]))
            for j in range(1, seam_matrix.shape[1]):
                temp = np.vstack((temp, image[i][seam_matrix[i][j - 1] + 1: seam_matrix[i][j] + 1]))
                temp = np.vstack((temp, image[i][seam_matrix[i][j]]))

            resize_image[i] = np.vstack((temp, image[i][seam_matrix[i][seam_matrix.shape[1] - 1] + 1:]))

    return resize_image


def resize(image: NDArray, out_height: int, out_width: int, forward_implementation: bool) -> Dict[str, NDArray]:
    """
    :param image: ِnp.array which represents an image.
    :param out_height: the resized image height
    :param out_width: the resized image width
    :param forward_implementation: a boolean flag that indicates whether forward or basic implementation is used.
                                    if forward_implementation is true then the forward-looking energy is used otherwise
                                    the basic implementation is used.
    :return: A dictionary with three elements, {'resized' : img1, 'vertical_seams' : img2 ,'horizontal_seams' : img3},
            where img1 is the resized image and img2/img3 are the visualization images
            (where the chosen seams are colored red and black for vertical and horizontal seams, respecitvely).
    """
    produce_images = list()
    resize_image = image.copy()
    k_height = image.shape[0] - out_height
    k_width = image.shape[1] - out_width
    ks = [(k_width, 'width', [255, 0, 0]), (k_height, 'height', [0, 0, 0])]  # color width seams with red and height seams with black

    start = time.time()
    for k, axis, color in ks:
        energy = get_gradients(resize_image)
        grayscale = to_grayscale(resize_image)
        color_matrix = np.ones_like(resize_image)
        index_matrix = np.tile(np.arange(resize_image.shape[1], dtype=int), (resize_image.shape[0], 1))
        seam_matrix = np.zeros(shape=(resize_image.shape[0], np.abs(k)), dtype=int)
        seam_counter = 0
        remove = True if k >= 0 else False
        for i in range(np.abs(k)):
            cost_matrix, path_matrix = forward_looking_energy_function(energy, grayscale, forward_implementation)
            # find seam
            entry = int(np.argmin(cost_matrix[-1]))
            optimal_seam = optimal_seam_backtrack(path_matrix, index_matrix, color_matrix, seam_matrix, color, entry, seam_counter)
            seam_counter += 1
            energy, grayscale, index_matrix = update_images(energy, grayscale, index_matrix, optimal_seam)

        produce_images.append(create_seams_image(resize_image, color_matrix, axis))
        seam_matrix = np.sort(seam_matrix, axis=1)
        resize_image = resize_image_by_seams(resize_image, seam_matrix, k, remove)
        if axis == 'width':
            resize_image = image_rotate(resize_image, 'left')
        if axis == 'height':
            resize_image = image_rotate(resize_image, 'right')
            produce_images[1] = image_rotate(produce_images[1], 'right')

    end = time.time()
    print(end-start)
    image_dict = {
        'resized': resize_image,
        'vertical_seams': produce_images[0],
        'horizontal_seams': produce_images[1]
    }
    return image_dict

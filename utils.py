from skimage import io
from scipy import ndimage
from torchvision import utils

import cv2
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import time
import torch


def order_points(pts):
    """
    Returns rectangle coordinates from quadrangle coordinates.

    :param pts: Coordinates of a quadrangle.
    :return: Numpy array of rectangle.
    """
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):
    """
    Returns a top-view of an image according to given points.

    :param image: Image, on which to apply affine transformation.
    :param pts: 4 corner coordinates of an image.
    :return: Image transformed with bird's view.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([[0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)

    # for i in range(pts.shape[0]):
    #     pts[i][0] = M[0][0] * pts[i][0] + M[1][0] * pts[i][1] + M[2][0]
    #     pts[i][1] = M[0][1] * pts[i][0] + M[1][1] * pts[i][1] + M[2][1]

    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def newline(p1, p2, color='blue'):
    """
    Draws a line on a matplotlib plot.

    :param p1: First x, y point of a line.
    :param p2: Second x, y point of a line.
    :param color: Color of a line to be drawn.
    """
    line = [tuple(p1), tuple(p2)]
    (line_xs, line_ys) = zip(*line)
    ax = plt.subplot()
    ax.add_line(plt.Line2D(line_xs, line_ys, color=color, lw=3))


def read_data(dataset_path="Dataset", ROI_data=False):
    """
    Reads data from ID cards or ROI dataset and returns it in dictionary format.

    :param dataset_path: Path to the dataset folder.
    :param ROI_data: If True, reads for ROI coordinates. Otherwise reads for ID card coordinates.
    :return: Dictionary of paths to images, corresponding key-point coordinates of ID cards and labels of classification
     or ROI coordinates with labels of Front or Back.
    """
    start = time.time()
    data = {}
    j = 0

    if ROI_data:
        for filename in os.listdir(dataset_path):  # Loop over the Dataset folder
            x, y = {}, {}
            for files in os.listdir(os.path.join(dataset_path, filename)):  # Loop over person's folder
                if files == 'Labels':
                    for label in os.listdir(
                            os.path.join(dataset_path, filename, files)):  # Loop over labels of the person's folder
                        with open(os.path.join(dataset_path, filename, files, label)) as json_file:
                            json_data = json.load(json_file)
                            for p in json_data["shapes"]:
                                if p['label'][0] == 'B':
                                    ID = 'B'
                                else:
                                    ID = 'F'
                                label, _ = os.path.splitext(label)
                                y[label] = {}
                                y[label]['label'] = ID
                                y[label]['points'] = {}
                                for i in range(5):
                                    try:
                                        if json_data['shapes'][i]['label'] == '{}-{}'.format(ID, i + 1):
                                            y[label]['points'].update(
                                                {json_data['shapes'][i]['label']: json_data['shapes'][i]['points']})
                                        else:
                                            y[label]['points'].update(
                                                {'{}-{}'.format(ID, i + 1): [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
                                                                             [0.0, 0.0]]})
                                    except IndexError:
                                        y[label]['points'].update(
                                            {'{}-{}'.format(ID, i + 1): [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
                                                                         [0.0, 0.0]]})
                                break
                else:
                    img_path = os.path.join(dataset_path, filename, files)
                    name, _ = os.path.splitext(files)
                    x.update({name: img_path})
            for i in x:
                data.update({j: {'image_path': x[i], 'label': y[i]['label'], 'points': y[i]['points']}})
                j += 1
    else:
        for filename in os.listdir(dataset_path):  # Loop over Dataset folder
            x, y = {}, {}
            if filename == "Classification_Dataset":
                for images in os.listdir(os.path.join(dataset_path, filename)):
                    img_path = os.path.join(dataset_path, filename, images)
                    zeros = np.zeros((1, 8))
                    zeros = np.asarray(zeros)
                    zeros = np.concatenate(zeros)
                    data.update({j: {'image_path': img_path, 'keypoints': zeros, 'label': [0]}})
                    j += 1
                continue
            else:
                for files in os.listdir(os.path.join(dataset_path, filename)):  # Loop over person's folder
                    if files == 'Labels':
                        for label in os.listdir(
                                os.path.join(dataset_path, filename, files)):  # Loop over labels of the person's folder
                            with open(os.path.join(dataset_path, filename, files, label)) as json_file:
                                json_data = json.load(json_file)
                                for p in json_data["shapes"]:
                                    label, _ = os.path.splitext(label)
                                    y.update({label: p["points"]})
                                    break
                    else:
                        img_path = os.path.join(dataset_path, filename, files)
                        name, _ = os.path.splitext(files)
                        x.update({name: img_path})
                for i in x:
                    y[i] = np.asarray(y[i])
                    y[i] = np.concatenate(y[i])
                    data.update({j: {'image_path': x[i], 'keypoints': y[i], 'label': [1]}})
                    j += 1

    end = time.time()
    print("Time elapsed to read data from {} (in seconds): {}".format(dataset_path, end - start))
    return data


def save_data(dataset="Resized_Dataset", ROI_data=False):
    """
    Given the dataset, resizes all of its images and readjusts their coordinates given in Labels folder and saves all
    changes.

    :param dataset: Path to the resized dataset folder.
    :param ROI_data: Whether to change the ID or ROI dataset.
    """
    start = time.time()
    print("Resizing images...")
    x, y = {}, {}
    j = 0

    for files in os.listdir(dataset):
        for images in os.listdir(os.path.join(dataset, files)):
            if images == "Labels":
                continue
            else:
                image = io.imread(os.path.join(dataset, files, images))
                h, w = image.shape[:2]
                image = cv2.resize(image, (224, 224))
                cv2.imwrite(os.path.join(dataset, files, images), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                image_path = os.path.join(dataset, files, images)
                temp, _ = os.path.splitext(images)
                label_path, _ = os.path.splitext(os.path.join(dataset, files, "Labels", temp))
                label_path = label_path + '.json'

                x.update({j: {'image_path': image_path, 'label_path': label_path, 'height': h, 'width': w}})
                print("Number of images resized: {}".format(j))
                j += 1

        j = 0
        print("Adjusting coordinates...")

        for i in range(len(x)):
            with open(x[i]["label_path"], 'r') as json_file:
                json_data = json.load(json_file)

            h, w = x[i]['height'], x[i]['width']

            if ROI_data:
                for index in range(5):
                    try:
                        json_data['shapes'][index]['points'][0][0] *= 224 / w
                        json_data['shapes'][index]['points'][0][1] *= 224 / h
                        json_data['shapes'][index]['points'][1][0] *= 224 / w
                        json_data['shapes'][index]['points'][1][1] *= 224 / h
                        json_data['shapes'][index]['points'][2][0] *= 224 / w
                        json_data['shapes'][index]['points'][2][1] *= 224 / h
                        json_data['shapes'][index]['points'][3][0] *= 224 / w
                        json_data['shapes'][index]['points'][3][1] *= 224 / h
                    except IndexError or KeyError:
                        continue
                    except TypeError:
                        json_data['shapes'][index]['points'][0] *= 224 / w
                        json_data['shapes'][index]['points'][1] *= 224 / h
                        json_data['shapes'][index]['points'][2] *= 224 / w
                        json_data['shapes'][index]['points'][3] *= 224 / h
                        json_data['shapes'][index]['points'][4] *= 224 / w
                        json_data['shapes'][index]['points'][5] *= 224 / h
                        json_data['shapes'][index]['points'][6] *= 224 / w
                        json_data['shapes'][index]['points'][7] *= 224 / h
            else:
                for p in json_data["shapes"]:
                    for k in range(4):
                        p["points"][k][0] *= 224 / w
                        p["points"][k][1] *= 224 / h

            with open(x[i]["label_path"], 'w') as json_file:
                json.dump(json_data, json_file)

            print("Number of labels adjusted: {}".format(j))
            j += 1

    end = time.time()
    print("Time elapsed to save data (in seconds): {}".format(end - start))


def model_check(net, ROI_data):
    """
    Visually shows the model's output on 1 of 15 random images.

    :param net: Neural network to be checked.
    :param ROI_data: Whether checking the ROI's or ID's model.
    """
    net.eval()
    with torch.no_grad():
        image = io.imread("Check/{}.jpg".format(np.random.randint(2, 3)))
        image_copy = image.copy()
        h, w = image_copy.shape[0], image_copy.shape[1]
        image = cv2.resize(image, (224, 224))
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = torch.FloatTensor(image).cuda()
        output = net(image)
        reg_out = output.cpu().numpy()

        if ROI_data:
            reg_out = array_to_coordinates(reg_out)

            for k in range(5):
                reg_out[k][0] *= w / 224
                reg_out[k][1] *= h / 224
                reg_out[k][2] *= w / 224
                reg_out[k][3] *= h / 224
                reg_out[k][4] *= w / 224
                reg_out[k][5] *= h / 224
                reg_out[k][6] *= w / 224
                reg_out[k][7] *= h / 224

        else:
            reg_out = keypoints_to_coordinates(reg_out)

            h, w = image_copy.shape[0], image_copy.shape[1]

            for k in range(4):
                reg_out[k][0] *= w / 224
                reg_out[k][1] *= h / 224

        show_image(image_copy, reg_out, ROI_data=ROI_data)


def coordinates_to_keypoints(coordinates):
    """
    Converts a 1x8 vector of coordinates into 4x2 array.

    :param coordinates: 1x8 vector of coordinates
    :return: 4x2 array of coordinates
    """
    result = []
    result.append([coordinates[0], coordinates[1]])
    result.append([coordinates[2], coordinates[3]])
    result.append([coordinates[4], coordinates[5]])
    result.append([coordinates[6], coordinates[7]])
    return result


def array_to_coordinates(keypoints):
    """
    Converts a 1x40 array of ROI coordinates into 5x8 array of coordinates.

    :param keypoints: 1x40 array of ROI coordinates.
    :return: 5x8 array of ROI coordinates.
    """
    result = np.zeros((5, 8))
    result[0] = keypoints[0][0:8]
    result[1] = keypoints[0][8:16]
    result[2] = keypoints[0][16:24]
    result[3] = keypoints[0][24:32]
    result[4] = keypoints[0][32:]
    return result


def keypoints_to_coordinates(keypoints, ROI_data=False):
    """
    Converts 1D vector of keypoints to 2D array of coordinates.

    :param keypoints: 1D vector of keypoints.
    :param ROI_data: If True, converts ROI keypoints. Otherwise converts ID keypoints.
    :return: Numpy 2D array of pairs of keypoints.
    """
    keypoints_copy = keypoints.copy()
    if ROI_data:
        polygon = np.zeros((5, 8))
        try:
            for i in range(5):
                keypoints_copy['F-{}'.format(i + 1)] = np.asarray(keypoints_copy['F-{}'.format(i + 1)])
                try:
                    keypoints_copy['F-{}'.format(i + 1)] = np.concatenate(keypoints_copy['F-{}'.format(i + 1)])
                except ValueError:
                    pass
                polygon[i] = keypoints_copy['F-{}'.format(i + 1)]
        except KeyError:
            for i in range(5):
                keypoints_copy['B-{}'.format(i + 1)] = np.asarray(keypoints_copy['B-{}'.format(i + 1)])
                try:
                    keypoints_copy['B-{}'.format(i + 1)] = np.concatenate(keypoints_copy['B-{}'.format(i + 1)])
                except ValueError:
                    pass
                polygon[i] = keypoints_copy['B-{}'.format(i + 1)]
    else:
        try:
            polygon = np.asarray([[keypoints_copy[0], keypoints_copy[1]],
                                  [keypoints_copy[2], keypoints_copy[3]],
                                  [keypoints_copy[4], keypoints_copy[5]],
                                  [keypoints_copy[6], keypoints_copy[7]]])
        except IndexError:
            polygon = np.asarray([[keypoints_copy[0][0], keypoints_copy[0][1]],
                                  [keypoints_copy[0][2], keypoints_copy[0][3]],
                                  [keypoints_copy[0][4], keypoints_copy[0][5]],
                                  [keypoints_copy[0][6], keypoints_copy[0][7]]])
    return polygon


def bird_view(image, pts, width=256):
    """
    Applying the top-view on an image, cropping it according to given coordinates and resizing with given width.

    :param image: Original image from which the object should be cropped.
    :param pts: Coordinates of an object to be cropped.
    :param width: Desired width of an object to be cropped. Height will be adjusted to it.
    :return: Image of an object in top-view (bird view).
    """
    temp = four_point_transform(image, pts)

    w, h = temp.shape[:2]
    r = width / float(w)
    dim = (int(h * r), width)
    temp = cv2.resize(temp, dim)

    return temp


def image_to_background(image, background):
    """
    Randomly putting the image to the background image.

    :param image: Foreground image.
    :param background: Background image.
    :return: Combined image with random placement of foreground image on background image and coordinates of 4 corners
    of an image along with translation coordinates x, y.
    """
    x = random.randint(0, background.shape[0] - image.shape[0])
    y = random.randint(0, background.shape[1] - image.shape[1])
    out = background

    rows, cols, channels = image.shape
    roi = background[x:x + rows, y:y + cols]

    img2gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    img2_fg = cv2.bitwise_and(image, image, mask=mask)

    dst = cv2.add(img1_bg, img2_fg)
    out[x:x + rows, y:y + cols] = dst
    coordinates = [[y, x], [y + cols, x], [y + cols, x + rows], [y, x + rows]]

    return out, coordinates, (x, y)


def split_data(coordinates):
    """
    Splits the 5x8 coordinates numpy array into 5 1x8 numpy arrays of coordinates.

    :param coordinates: 5x8 numpy array of coordinates
    :return: 5 1x8 numpy arrays of coordinates
    """
    data_1 = coordinates[0]
    data_2 = coordinates[1]
    data_3 = coordinates[2]
    data_4 = coordinates[3]
    data_5 = coordinates[4]
    return data_1, data_2, data_3, data_4, data_5


def show_keypoints(image, keypoints, ROI_data=False):
    """
    Drawing keypoints on a matplotlib plot.

    :param image: Image that contains an ID card.
    :param keypoints: 4 keypoints of an ID card or 20 keypoints of ROI.
    :param ROI_data: If True, draws ROI keypoints. Otherwise draws ID card keypoints.
    """
    plt.imshow(image)
    if ROI_data:
        try:
            if keypoints[0][6] is None:
                for i in range(5):
                    for j in range(4):
                        plt.scatter(keypoints[i][j][0], keypoints[i][j][1], s=10, marker='.', c='r')
            else:
                for i in range(5):
                    for j in range(0, 8, 2):
                        plt.scatter(keypoints[i][j], keypoints[i][j + 1], s=10, marker='.', c='r')
        except IndexError:
            for i in range(5):
                for j in range(4):
                    plt.scatter(keypoints[i][j][0], keypoints[i][j][1], s=10, marker='.', c='r')
    else:
        for i in range(0, 8, 2):
            plt.scatter(keypoints[i], keypoints[i + 1], s=10, marker='.', c='r')
    plt.pause(0.001)


def show_image(image, keypoints, ROI_data=False):
    """
    Shows an image along with its keypoints drawn using lines.

    :param image: Image to be shown.
    :param keypoints: Keypoints of an ID card or ROIs in the image to be shown.
    :param ROI_data: If True, draws ROI lines on the image. Otherwise draws ID card lines.
    """
    if ROI_data:
        plt.figure()
        show_keypoints(image, keypoints, ROI_data=True)
        try:
            if keypoints[0][6] is None:
                for i in range(5):
                    newline(keypoints[i][0], keypoints[i][1], color='red')
                    newline(keypoints[i][1], keypoints[i][2], color='red')
                    newline(keypoints[i][2], keypoints[i][3], color='red')
                    newline(keypoints[i][3], keypoints[i][0], color='red')
            else:
                for i in range(5):
                    newline(keypoints[i][:2], keypoints[i][2:4], color='red')
                    newline(keypoints[i][2:4], keypoints[i][4:6], color='red')
                    newline(keypoints[i][4:6], keypoints[i][6:], color='red')
                    newline(keypoints[i][6:], keypoints[i][:2], color='red')
        except IndexError:
            for i in range(5):
                newline(keypoints[i][0], keypoints[i][1], color='red')
                newline(keypoints[i][1], keypoints[i][2], color='red')
                newline(keypoints[i][2], keypoints[i][3], color='red')
                newline(keypoints[i][3], keypoints[i][0], color='red')
    else:
        try:
            keypoints = keypoints_to_coordinates(keypoints)
        except IndexError:
            pass
        p1 = keypoints[0]
        p2 = keypoints[1]
        p3 = keypoints[2]
        p4 = keypoints[3]

        keypoints = np.asarray(keypoints)
        keypoints = np.concatenate(keypoints)

        plt.figure()
        show_keypoints(image, keypoints)
        newline(p1, p2, color='red')
        newline(p2, p3, color='red')
        newline(p3, p4, color='red')
        newline(p4, p1, color='red')
    plt.show()


def color_jitter(image):
    """
    Applies a random color jitter on an image.

    :param image: Image on which color jitter should be applied.
    :return: Image with random color jitter applied on it.
    """
    h, w, c = image.shape
    zitter = np.zeros_like(image)
    zitter[:, :, 0] = np.random.randint(0, random.randint(1, 50), (h, w))
    zitter[:, :, 1] = np.random.randint(0, random.randint(1, 50), (h, w))
    zitter[:, :, 2] = np.random.randint(0, random.randint(1, 50), (h, w))
    image = cv2.add(image, zitter)
    return image


def rotate_coordinates(x, y, xm, ym, angle):
    """
    Rotates x and y coordinates by the given angle and by given center of rotation coordinates.

    :param x: Initial x coordinate.
    :param y: Initial y coordinate.
    :param xm: X coordinate of a center of rotation.
    :param ym: Y coordinate of a center of rotation.
    :param angle: Angle of rotation in degrees.
    :return: X and y coordinates after rotation.
    """
    angle = -math.radians(angle)

    xr = (x - xm) * math.cos(angle) - (y - ym) * math.sin(angle) + xm
    yr = (x - xm) * math.sin(angle) + (y - ym) * math.cos(angle) + ym

    return xr, yr


def rotate(image, angle):
    """
    Rotates an image by the given angle.

    :param image: Image to be rotated.
    :param angle: Angle in degrees by which to rotate the image.
    :return: Rotated image.
    """
    image = ndimage.rotate(image, angle)
    return image


def centroid(vertexes):
    """
    Finds x and y coordinates of a center given a list of vertices.

    :param vertexes: List of x, y coordinates of vertices of a polygon.
    :return: X and y coordinates of a centre of a given polygon.
    """
    x_list = [vertex[0] for vertex in vertexes]
    y_list = [vertex[1] for vertex in vertexes]
    _len = len(vertexes)
    x = sum(x_list) / _len
    y = sum(y_list) / _len
    return x, y


def augment_data(IDcard_image, IDcard_keypoints, num):
    """
    Creates 1 augmented data from the given ID card image and keypoints.

    :param IDcard_image: Photo of an ID card.
    :param IDcard_keypoints: Coordinates of ID card's corners.
    :param num: Background number.
    :return: Image that contains a combination of background and ID card image with random color jitter, rotation,
    background crop, translation and resize along with readjusted ID card keypoints.
    """

    random_angle = np.random.randint(-30, 30)

    # Color Jitter

    IDcard_image = color_jitter(IDcard_image)

    # Mask

    affine_pts = np.array(IDcard_keypoints, dtype=np.int)
    rect = cv2.boundingRect(affine_pts)
    x, y, w, h = rect
    cropped = IDcard_image[y:y + h, x:x + w].copy()

    affine_pts = affine_pts - affine_pts.min(axis=0)

    mask = np.zeros(cropped.shape[:2], np.uint8)
    cv2.drawContours(mask, [affine_pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
    dst = cv2.bitwise_and(cropped, cropped, mask=mask)
    bg = np.ones_like(cropped, np.uint8) * 255
    cv2.bitwise_not(bg, bg, mask=mask)

    # Crop

    background = cv2.imread("Background/{}.jpg".format(num))

    random_x = np.random.randint(0, int(background.shape[1] / 2))
    random_y = np.random.randint(0, int(background.shape[0] / 2))

    background = background[random_x:random_x + int(background.shape[1] / 2),
                 random_y:random_y + int(background.shape[0] / 2)]

    # Resize

    random_size = (np.random.random_sample() + 0.3) / 2 * background.shape[1]

    h, w = dst.shape[:2]
    r = random_size / float(h)
    dim = (int(w * r), int(random_size))
    print(dim)
    dst = cv2.resize(dst, dim)

    affine_pts = affine_pts.astype(float)

    for i in range(4):
        affine_pts[i][0] *= random_size / h
        affine_pts[i][1] *= int(w * r) / w

    # Rotate

    center_x_1 = dst.shape[1] / 2
    center_y_1 = dst.shape[0] / 2

    dst = rotate(dst, random_angle)

    center_y_2 = dst.shape[0] / 2
    center_x_2 = dst.shape[1] / 2

    center_y_diff = center_y_2 - center_y_1
    center_x_diff = center_x_2 - center_x_1

    for i in range(4):
        affine_pts[i][0] += center_x_diff
        affine_pts[i][1] += center_y_diff

    center_y_2, center_x_2 = centroid(affine_pts)

    affine_pts[0] = rotate_coordinates(affine_pts[0][0], affine_pts[0][1], center_y_2, center_x_2, random_angle)
    affine_pts[1] = rotate_coordinates(affine_pts[1][0], affine_pts[1][1], center_y_2, center_x_2, random_angle)
    affine_pts[2] = rotate_coordinates(affine_pts[2][0], affine_pts[2][1], center_y_2, center_x_2, random_angle)
    affine_pts[3] = rotate_coordinates(affine_pts[3][0], affine_pts[3][1], center_y_2, center_x_2, random_angle)

    # Background

    while dst.shape[0] > background.shape[0] or dst.shape[1] > background.shape[1]:
        key = random.randint(1, 50)
        background = cv2.imread("Background/{}.jpg".format(key))

    background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

    output, points, (translate_x, translate_y) = image_to_background(dst, background)

    # Translate

    for i in range(4):
        affine_pts[i][0] += translate_y
        affine_pts[i][1] += translate_x

    return output, affine_pts


def show_landmarks_batch(sample_batch, predictions_batch):
    """
    Plots 4 ID card images with corresponding ID card keypoints drawn using lines with matplotlib.

    :param sample_batch: Batch of 4 images along with their keypoints to be shown.
    :param predictions_batch: Batch of 4 keypoint predictions for images.
    """
    images_batch, keypoints_batch = sample_batch['image'].cpu(), sample_batch['keypoints'].cpu()

    predictions_batch = predictions_batch.cpu()
    im_size = images_batch.shape[2]
    grid_border_size = 2

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(4):
        plt.scatter(predictions_batch[i, 0].numpy() + i * im_size + (i + 1) * grid_border_size,
                    predictions_batch[i, 1].numpy() + grid_border_size,
                    s=100, marker='.', c='r')
        plt.scatter(predictions_batch[i, 2].numpy() + i * im_size + (i + 1) * grid_border_size,
                    predictions_batch[i, 3].numpy() + grid_border_size,
                    s=100, marker='.', c='r')
        plt.scatter(predictions_batch[i, 4].numpy() + i * im_size + (i + 1) * grid_border_size,
                    predictions_batch[i, 5].numpy() + grid_border_size,
                    s=100, marker='.', c='r')
        plt.scatter(predictions_batch[i, 6].numpy() + i * im_size + (i + 1) * grid_border_size,
                    predictions_batch[i, 7].numpy() + grid_border_size,
                    s=100, marker='.', c='r')
        p1 = [predictions_batch[i, 0].numpy() + i * im_size + (i + 1) * grid_border_size,
              predictions_batch[i, 1].numpy() + grid_border_size]
        p2 = [predictions_batch[i, 2].numpy() + i * im_size + (i + 1) * grid_border_size,
              predictions_batch[i, 3].numpy() + grid_border_size]
        p3 = [predictions_batch[i, 4].numpy() + i * im_size + (i + 1) * grid_border_size,
              predictions_batch[i, 5].numpy() + grid_border_size]
        p4 = [predictions_batch[i, 6].numpy() + i * im_size + (i + 1) * grid_border_size,
              predictions_batch[i, 7].numpy() + grid_border_size]
        newline(p1, p2, color='red')
        newline(p2, p3, color='red')
        newline(p3, p4, color='red')
        newline(p4, p1, color='red')
        plt.scatter(keypoints_batch[i, 0].numpy() + i * im_size + (i + 1) * grid_border_size,
                    keypoints_batch[i, 1].numpy() + grid_border_size,
                    s=100, marker='.', c='b')
        plt.scatter(keypoints_batch[i, 2].numpy() + i * im_size + (i + 1) * grid_border_size,
                    keypoints_batch[i, 3].numpy() + grid_border_size,
                    s=100, marker='.', c='b')
        plt.scatter(keypoints_batch[i, 4].numpy() + i * im_size + (i + 1) * grid_border_size,
                    keypoints_batch[i, 5].numpy() + grid_border_size,
                    s=100, marker='.', c='b')
        plt.scatter(keypoints_batch[i, 6].numpy() + i * im_size + (i + 1) * grid_border_size,
                    keypoints_batch[i, 7].numpy() + grid_border_size,
                    s=100, marker='.', c='b')
        p5 = [keypoints_batch[i, 0].numpy() + i * im_size + (i + 1) * grid_border_size,
              keypoints_batch[i, 1].numpy() + grid_border_size]
        p6 = [keypoints_batch[i, 2].numpy() + i * im_size + (i + 1) * grid_border_size,
              keypoints_batch[i, 3].numpy() + grid_border_size]
        p7 = [keypoints_batch[i, 4].numpy() + i * im_size + (i + 1) * grid_border_size,
              keypoints_batch[i, 5].numpy() + grid_border_size]
        p8 = [keypoints_batch[i, 6].numpy() + i * im_size + (i + 1) * grid_border_size,
              keypoints_batch[i, 7].numpy() + grid_border_size]
        newline(p5, p6)
        newline(p6, p7)
        newline(p7, p8)
        newline(p8, p5)

        plt.title('Batch from dataloader')


def num_flat_features(x):
    """
    Flatten layer for x.

    :param x: Result of a CNN that needs to be flattened.
    :return: Number of features that x has.
    """
    size = x.size()[1:]
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

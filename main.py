from ID import Identity, Net
from pytesseract import image_to_string
from ROI import FrontROI, BackROI
from skimage import io
from utils import keypoints_to_coordinates, array_to_coordinates, split_data, coordinates_to_keypoints, bird_view, show_image

import cv2
import numpy as np
import re
import torch
import torchvision


def run(image_path, resnet, front=True):
    """
    Given the image of the ID card, produces the JSON file with all the fields inside it.

    :param image_path: Path to the image of an ID card.
    :param front: Whether the image of an ID card is front or back.
    :return: String text with all the fields of the front or back side of an ID card.
    """
    config_block = "-l rus --oem 1 --psm 6"
    config_row = "-l rus --oem 1 --psm 7"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image = io.imread(image_path)

    # ID card classification

    # resnet = torchvision.models.resnet152(pretrained=True, progress=True).cuda()
    resnet.avgpool = Identity()
    resnet.fc = Identity()

    net = Net(resnet)
    net.to(device)
    net.load_state_dict(torch.load("IDclassification.pt"))

    net.eval()
    image_copy = image.copy()
    image = cv2.resize(image, (224, 224))
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, 0)
    image = torch.FloatTensor(image).cuda()
    IDoutput, _ = net(image)
    IDoutput = IDoutput.item()

    if IDoutput < 0.5:
        return 'ERROR! No ID card was detected! (Or photo is rotated)'

    # ID card regression

    net = Net(resnet)
    net.to(device)
    net.load_state_dict(torch.load("IDregression.pt"))

    net.eval()
    with torch.no_grad():
        image = cv2.resize(image_copy, (224, 224))
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = torch.FloatTensor(image).cuda()
        _, IDoutput = net(image)
        IDoutput = IDoutput.cpu().numpy()
        IDoutput = keypoints_to_coordinates(IDoutput)

        h, w = image_copy.shape[0], image_copy.shape[1]

        for k in range(4):
            IDoutput[k][0] *= w / 224
            IDoutput[k][1] *= h / 224

        affine_pts = np.array(IDoutput, dtype=np.int)
        rect = cv2.boundingRect(affine_pts)
        x, y, w, h = rect
        cropped = image_copy[y:y + h, x:x + w].copy()
        image = cv2.bitwise_and(cropped, cropped)

    # ROI regression

    if front:
        ROINet = FrontROI(resnet)
        ROINet.to(device)
        ROINet.load_state_dict(torch.load("FrontROI.pt"))
    else:
        ROINet = BackROI(resnet)
        ROINet.to(device)
        ROINet.load_state_dict(torch.load("BackROI.pt"))

    ROINet.eval()
    with torch.no_grad():
        image_copy = image.copy()
        h, w = image_copy.shape[0], image_copy.shape[1]
        image = cv2.resize(image, (224, 224))
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = torch.FloatTensor(image)
        image = image.to(device)

        ROIoutput = ROINet(image)
        ROIoutput = ROIoutput.cpu().numpy()
        ROIoutput = array_to_coordinates(ROIoutput)

    for k in range(5):
        ROIoutput[k][0] *= w / 224
        ROIoutput[k][1] *= h / 224
        ROIoutput[k][2] *= w / 224
        ROIoutput[k][3] *= h / 224
        ROIoutput[k][4] *= w / 224
        ROIoutput[k][5] *= h / 224
        ROIoutput[k][6] *= w / 224
        ROIoutput[k][7] *= h / 224

    row1, row2, row3, row4, row5 = split_data(ROIoutput)

    row1 = coordinates_to_keypoints(row1)
    row2 = coordinates_to_keypoints(row2)
    row3 = coordinates_to_keypoints(row3)
    row4 = coordinates_to_keypoints(row4)
    row5 = coordinates_to_keypoints(row5)

    # Tesseract OCR

    # data = {}

    if front:
        affine_pts = np.array(row1, dtype=np.int)
        dst = bird_view(image_copy, affine_pts)
        line1 = image_to_string(dst, config=config_block)
        line1 = re.sub(r'[^А-я]+', '', line1)
        line1 = line1.upper()
        line1 = line1[line1.find("ЛИЯ") + 3:]
        # data.update({'Фамилия': line1})
        text1 = "Фамилия: {}\n".format(line1)

        affine_pts = np.array(row2, dtype=np.int)
        dst = bird_view(image_copy, affine_pts)
        line2 = image_to_string(dst, config=config_block)
        line2 = re.sub(r'[^А-я]+', '', line2)
        line2 = line2.upper()
        line2 = line2[line2.find("ИМЯ") + 3:]
        # data.update({'Имя': line2})
        text2 = "Имя: {}\n".format(line2)

        affine_pts = np.array(row3, dtype=np.int)
        dst = bird_view(image_copy, affine_pts)
        line3 = image_to_string(dst, config=config_block)
        line3 = re.sub(r'[^А-я]+', '', line3)
        line3 = line3.upper()
        line3 = line3[line3.find("СТВО") + 4:]
        # data.update({'Отчество': line3})
        text3 = "Отчество: {}\n".format(line3)

        affine_pts = np.array(row4, dtype=np.int)
        dst = bird_view(image_copy, affine_pts)
        line4 = image_to_string(dst, config=config_block)
        line4 = re.sub('/[^.0-9]+/i', '', line4)
        line4 = line4[line4.find("ЖДЕНИЯ\n") + 7:]
        # data.update({'Дата рождения': line4})
        text4 = "Дата рождения: {}\n".format(line4)

        affine_pts = np.array(row5, dtype=np.int)
        dst = bird_view(image_copy, affine_pts)
        line5 = image_to_string(dst, config=config_row)
        line5 = re.sub('[^0-9]+', '', line5)
        # data.update({'ИИН': line5})
        text5 = "ИИН: {}".format(line5)
        text6 = ""
    else:
        affine_pts = np.array(row1, dtype=np.int)
        dst = bird_view(image_copy, affine_pts)
        line1 = image_to_string(dst, config=config_block)
        line1 = re.sub('[^0-9]+', '', line1)
        # data.update({'Номер удостоверения личности': line1})
        text1 = "Номер удостоверения личности: {}\n".format(line1)

        affine_pts = np.array(row2, dtype=np.int)
        dst = bird_view(image_copy, affine_pts)
        line2 = image_to_string(dst, config=config_block)
        line2 = re.sub(r'[^-.А-я ]+', '', line2)
        line2 = line2.upper()
        line2 = line2[line2.find("ЖДЕНИЯ") + 6:]
        # data.update({'Место рождения': line2})
        text2 = "Место рождения: {}\n".format(line2)

        affine_pts = np.array(row3, dtype=np.int)
        dst = bird_view(image_copy, affine_pts)
        line3 = image_to_string(dst, config=config_block)
        line3 = re.sub(r'[^А-я]+', '', line3)
        line3 = line3.upper()
        line3 = line3[line3.find("ОСТЬ") + 4:]
        # data.update({'Национальность': line3})
        text3 = "Национальность: {}\n".format(line3)

        affine_pts = np.array(row4, dtype=np.int)
        dst = bird_view(image_copy, affine_pts)
        line4 = image_to_string(dst, config=config_block)
        line4 = re.sub(r'[^А-я ]+', '', line4)
        line4 = line4.upper()
        line4 = line4[line4.find("МВД"):]
        # data.update({'Орган выдачи': line4})
        text4 = "Орган выдачи: {}\n".format(line4)

        affine_pts = np.array(row5, dtype=np.int)
        dst = bird_view(image_copy, affine_pts)
        line5 = image_to_string(dst, config=config_block)
        line5 = re.sub('/[^.0-9 ]+/i', '', line5)
        line5 = line5[line5.find("\n") + 1:]
        line6 = line5[11:]
        line5 = line5[:10]
        line6 = line6[:10]
        # data.update({'Дата выдачи': line5})
        # data.update({'Срок действия': line6})
        text5 = "Дата выдачи: {}\n".format(line5)
        text6 = "Срок действия: {}".format(line6)

    # with open(file_path, 'w', encoding='utf-8') as f:
    #     json.dump(data, f, indent=4, ensure_ascii=False)

    text = text1 + text2 + text3 + text4 + text5 + text6
    return text


# if __name__ == "__main__":
#     resnet = torchvision.models.resnet152(pretrained=True, progress=True).cuda()
#     run('Check/1.jpg', resnet, front=False)

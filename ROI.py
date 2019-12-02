from torch.utils.data import Dataset
from skimage import io
from utils import read_data, num_flat_features

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ROIDataset(Dataset):
    """
    Dataset of ROI images in ID cards.
    """

    def __init__(self, dataset_path, transform=None):
        self.data = read_data(dataset_path, ROI_data=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        temp = self.data[idx]
        image = io.imread(temp['image_path'])
        sample = {'image': image, 'keypoints': temp['points'], 'label': temp['label']}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """
    Class that defines a method for images of ID cards for converting them into Tensors.
    """

    def __call__(self, sample):
        points = []
        image = sample['image']
        for k in range(1, 6):
            points.append(sample['keypoints']['{}-{}'.format(sample['label'], k)])
        points = np.concatenate(points)
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)
        return {'image': image,
                'keypoints': torch.FloatTensor(points),
                'label': sample['label']}


class FrontROI(nn.Module):
    """
    CNN with ResNet152 backbone for finding ROI keypoints of a front of an ID card.
    """

    def __init__(self, resnet):
        super(FrontROI, self).__init__()

        self.resnet = resnet

        self.fc1 = nn.Linear(100352, 120)
        self.fc2 = nn.Linear(120, 84)

        self.fc_reg = nn.Linear(84, 40)

    def forward(self, x):
        x = self.resnet(x)

        x = x.view(-1, num_flat_features(x))

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x_reg = self.fc_reg(x)

        return x_reg


class BackROI(nn.Module):
    """
    CNN with ResNet152 backbone for finding ROI keypoints of a back of an ID card.
    """

    def __init__(self, resnet):
        super(BackROI, self).__init__()

        self.resnet = resnet

        self.fc1 = nn.Linear(100352, 120)
        self.fc2 = nn.Linear(120, 84)

        self.fc_reg = nn.Linear(84, 40)

    def forward(self, x):
        x = self.resnet(x)

        x = x.view(-1, num_flat_features(x))

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x_reg = self.fc_reg(x)

        return x_reg


class Identity(nn.Module):
    """
    Identity module for changing the output of a ResNet152 backbone.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x

# if __name__ == "__main__":

# Step 1. Loading the data and transforming it into tensor

# while True:
#
#     key = np.random.randint(1, 300)
#
#     image = cv2.imread("Clean_Front_ROI_Dataset_Resized/Testing Data/Images/{}.jpg".format(key))
#     ROI_coordinates = []
#     with open("Clean_Front_ROI_Dataset_Resized/Testing Data/Images/Labels/{}.json".format(key), 'r') as json_file:
#         json_data = json.load(json_file)
#         ROI_coordinates.append(json_data['shapes'][0]['points'])
#         ROI_coordinates.append(json_data['shapes'][1]['points'])
#         ROI_coordinates.append(json_data['shapes'][2]['points'])
#         ROI_coordinates.append(json_data['shapes'][3]['points'])
#         ROI_coordinates.append(json_data['shapes'][4]['points'])
#
#         print("ROI label: {}".format(json_data['shapes'][0]['label']))
#
#         print("Random key: {}".format(key))
#
#     print("ROI coordinates: {}".format(ROI_coordinates))
#
#     show_image(image, ROI_coordinates, ROI_data=True)
#     print("STEP 2 - SUCCESS! Sample image was plotted!")

# Step 2. Saving images in 224x224 size and adjusting their ROI coordinates

# save_data("ROI_Resized_Dataset", ROI_data=True)

# Step 3. Augmenting ROI data

# config = "-l rus --oem 1 --psm 7"
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# train_dataset = ROIDataset("Clean_Back_ROI_Dataset_Resized/Training Data", transform=ToTensor())
# test_dataset = ROIDataset("Clean_Back_ROI_Dataset_Resized/Testing Data", transform=ToTensor())
#
# trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
# validloader = torch.utils.data.DataLoader(test_dataset, batch_size=4)
#
# print("Size of training set:", len(train_dataset))
# print("Size of testing set:", len(test_dataset))

# Step 2. Plotting a sample from the dataset

# dataset = "Clean_Front_ROI_Dataset_Resized"

# resnet = torchvision.models.resnet152(pretrained=True, progress=True).cuda()
# resnet.avgpool = Identity()
# resnet.fc = Identity()
#
# backNet = FrontROI(resnet)
# backNet.to(device)
# backNet.load_state_dict(torch.load("FrontROI2.pt"))
#
# criterion_cls = nn.BCELoss()
# criterion_reg = nn.MSELoss()
# optimizer = torch.optim.Adam(backNet.parameters(), lr=0.0001)

# total = 0
# removed = 0
#
# net.eval()
#
# for files in os.listdir(dataset):
#     for folder in os.listdir(os.path.join(dataset, files)):
#         for filename in os.listdir(os.path.join(dataset, files, folder)):
#             if filename == "Labels":
#                 continue
#             image = io.imread(os.path.join(dataset, files, folder, filename))
#             image_copy = image.copy()
#             h, w = image_copy.shape[0], image_copy.shape[1]
#             image = cv2.resize(image, (224, 224))
#             image = image.transpose((2, 0, 1))
#             image = np.expand_dims(image, 0)
#             image = torch.FloatTensor(image).cuda()
#             output, _ = net(image)
#             output = output.item()
#             label, _ = os.path.splitext(filename)
#
#             total += 1
#
#             if output < 0.5:
#                 os.remove(os.path.join(dataset, files, folder, filename))
#                 removed += 1
#
#                 print('*' * 250)
#                 print("Removed images: {}".format(removed))
#                 print('*' * 250)
#
#             print("Images checked: {}".format(total))
#
# print("Front dataset done!")
#
# dataset = "Clean_Back_ROI_Dataset_Resized"
#
# total = 0
# removed = 0
#
# for files in os.listdir(dataset):
#     for folder in os.listdir(os.path.join(dataset, files)):
#         for filename in os.listdir(os.path.join(dataset, files, folder)):
#             if filename == "Labels":
#                 continue
#             image = io.imread(os.path.join(dataset, files, folder, filename))
#             image_copy = image.copy()
#             h, w = image_copy.shape[0], image_copy.shape[1]
#             image = cv2.resize(image, (224, 224))
#             image = image.transpose((2, 0, 1))
#             image = np.expand_dims(image, 0)
#             image = torch.FloatTensor(image).cuda()
#             output, _ = net(image)
#             output = output.item()
#             label, _ = os.path.splitext(filename)
#
#             total += 1
#
#             if output < 0.5:
#                 os.remove(os.path.join(dataset, files, folder, filename))
#                 removed += 1
#
#                 print('*' * 250)
#                 print("Removed images: {}".format(removed))
#                 print('*' * 250)
#
#             print("Images checked: {}".format(total))
#
# print("Back dataset done!")
#
# name_list = []
#
# for files in os.listdir("Clean_Front_ROI_Dataset_Resized/Training Data/Images"):
#     filename, _ = os.path.splitext(files)
#     name_list.append(filename)
#
# for files in os.listdir("Clean_Front_ROI_Dataset_Resized/Training Data/Images/Labels"):
#     filename, _ = os.path.splitext(files)
#     if filename in name_list:
#         continue
#     else:
#         os.remove("Clean_Front_ROI_Dataset_Resized/Training Data/Images/Labels/{}.json".format(filename))
#
# name_list = []
#
# for files in os.listdir("Clean_Front_ROI_Dataset_Resized/Testing Data/Images"):
#     filename, _ = os.path.splitext(files)
#     name_list.append(filename)
#
# for files in os.listdir("Clean_Front_ROI_Dataset_Resized/Testing Data/Images/Labels"):
#     filename, _ = os.path.splitext(files)
#     if filename in name_list:
#         continue
#     else:
#         os.remove("Clean_Front_ROI_Dataset_Resized/Testing Data/Images/Labels/{}.json".format(filename))
#
# name_list = []
#
# for files in os.listdir("Clean_Back_ROI_Dataset_Resized/Training Data/Images"):
#     filename, _ = os.path.splitext(files)
#     name_list.append(filename)
#
# for files in os.listdir("Clean_Back_ROI_Dataset_Resized/Training Data/Images/Labels"):
#     filename, _ = os.path.splitext(files)
#     if filename in name_list:
#         continue
#     else:
#         os.remove("Clean_Back_ROI_Dataset_Resized/Training Data/Images/Labels/{}.json".format(filename))
#
# name_list = []
#
# for files in os.listdir("Clean_Back_ROI_Dataset_Resized/Testing Data/Images"):
#     filename, _ = os.path.splitext(files)
#     name_list.append(filename)
#
# for files in os.listdir("Clean_Back_ROI_Dataset_Resized/Testing Data/Images/Labels"):
#     filename, _ = os.path.splitext(files)
#     if filename in name_list:
#         continue
#     else:
#         os.remove("Clean_Back_ROI_Dataset_Resized/Testing Data/Images/Labels/{}.json".format(filename))
#
# print("Success!")

# while True:
#
#     key = np.random.randint(1, 146)
#
#     backNet.eval()
#     with torch.no_grad():
#
#         with open("ID_Dataset_Original/Till 500/Labels/{}.json".format(key), 'r') as json_file:
#             json_data = json.load(json_file)
#             for p in json_data['shapes']:
#                 output = p['points']
#
#         image = io.imread("ID_Dataset_Original/Till 500/{}.jpg".format(key))
#         image_copy = image.copy()
#         h, w = image_copy.shape[0], image_copy.shape[1]
#         image = cv2.resize(image, (224, 224))
#         image = image.transpose((2, 0, 1))
#         image = np.expand_dims(image, 0)
#         image = torch.FloatTensor(image)
#         image = image.to(device)

# affine_pts = np.array(output, dtype=np.int)
# rect = cv2.boundingRect(affine_pts)
# x, y, w, h = rect
# cropped = image[y:y + h, x:x + w].copy()
# image_copy = cv2.bitwise_and(cropped, cropped)
# output = np.asarray(output)

# h, w = image_copy.shape[0], image_copy.shape[1]
# image = cv2.resize(image, (224, 224))
# image = image.transpose((2, 0, 1))
# image = np.expand_dims(image, 0)
# image = torch.FloatTensor(image).cuda()
# reg_out = backNet(image)
# reg_out = reg_out.cpu().numpy()
# reg_out = array_to_coordinates(reg_out)
#
# for k in range(5):
#     reg_out[k][0] *= w / 224
#     reg_out[k][1] *= h / 224
#     reg_out[k][2] *= w / 224
#     reg_out[k][3] *= h / 224
#     reg_out[k][4] *= w / 224
#     reg_out[k][5] *= h / 224
#     reg_out[k][6] *= w / 224
#     reg_out[k][7] *= h / 224
#
# B1, B2, B3, B4, B5 = split_data(reg_out)
#
# show_image(image_copy, reg_out, ROI_data=True)
# B1 = coordinates_to_keypoints(B1)
# B2 = coordinates_to_keypoints(B2)
# B3 = coordinates_to_keypoints(B3)
# B4 = coordinates_to_keypoints(B4)
# B5 = coordinates_to_keypoints(B5)
#
# affine_pts = np.array(B1, dtype=np.int)
# rect = cv2.boundingRect(affine_pts)
# x, y, w, h = rect
# cropped = image_copy[y:y + h, x:x + w].copy()
# cropped = cv2.bitwise_and(cropped, cropped)
# dst = bird_view(image_copy, affine_pts)
#
# affine_pts = np.concatenate(affine_pts)
# affine_pts = np.expand_dims(affine_pts, axis=0)
# show_image(dst, affine_pts)
#
# print("First row: {}".format(image_to_string(dst, config=config)))
#
# affine_pts = np.array(B2, dtype=np.int)
# rect = cv2.boundingRect(affine_pts)
# x, y, w, h = rect
# cropped = image_copy[y:y + h, x:x + w].copy()
# cropped = cv2.bitwise_and(cropped, cropped)
# dst = bird_view(image_copy, affine_pts)
#
# affine_pts = np.concatenate(affine_pts)
# affine_pts = np.expand_dims(affine_pts, axis=0)
# show_image(dst, affine_pts)
#
# print("Second row: {}".format(image_to_string(dst, config=config)))
#
# affine_pts = np.array(B3, dtype=np.int)
# rect = cv2.boundingRect(affine_pts)
# x, y, w, h = rect
# cropped = image_copy[y:y + h, x:x + w].copy()
# cropped = cv2.bitwise_and(cropped, cropped)
# dst = bird_view(image_copy, affine_pts)
#
# affine_pts = np.concatenate(affine_pts)
# affine_pts = np.expand_dims(affine_pts, axis=0)
# show_image(dst, affine_pts)
#
# print("Third row: {}".format(image_to_string(dst, config=config)))
#
# affine_pts = np.array(B4, dtype=np.int)
# rect = cv2.boundingRect(affine_pts)
# x, y, w, h = rect
# cropped = image_copy[y:y + h, x:x + w].copy()
# cropped = cv2.bitwise_and(cropped, cropped)
# dst = bird_view(image_copy, affine_pts)
#
# affine_pts = np.concatenate(affine_pts)
# affine_pts = np.expand_dims(affine_pts, axis=0)
# show_image(dst, affine_pts)
#
# print("Fourth row: {}".format(image_to_string(dst, config=config)))
#
# affine_pts = np.array(B5, dtype=np.int)
# rect = cv2.boundingRect(affine_pts)
# x, y, w, h = rect
# cropped = image_copy[y:y + h, x:x + w].copy()
# cropped = cv2.bitwise_and(cropped, cropped)
# dst = bird_view(image_copy, affine_pts)
#
# affine_pts = np.concatenate(affine_pts)
# affine_pts = np.expand_dims(affine_pts, axis=0)
# show_image(dst, affine_pts)
#
# print("Fifth row: {}".format(image_to_string(dst, config=config)))

# resnet = torchvision.models.resnet152(pretrained=True, progress=True).cuda()
# resnet.avgpool = Identity()
# resnet.fc = Identity()
#
# frontNet = FrontROI(resnet)
# frontNet.to(device)
# frontNet.load_state_dict(torch.load("FrontROI.pt"))

# criterion_reg = nn.MSELoss()
# optimizer = torch.optim.Adam(frontNet.parameters(), lr=0.0001)

# min_valid = 10000
# #
# # # model_check(frontNet, ROI_data=True)
# #
# for epoch in range(100):
#
#     backNet.train()
#     running_loss = 0.0
#     valid_loss = 0.0
#
#     print("Epoch: {}".format(epoch + 1))
#
#     for i, data in enumerate(trainloader):
#
#         inputs, keypoints = data['image'].to(device, dtype=torch.float), data['keypoints'].to(device)
#         optimizer.zero_grad()
#
#         reg_out = backNet(inputs)
#         loss_reg = criterion_reg(reg_out, keypoints)
#         loss_reg.backward()
#         optimizer.step()
#         running_loss += loss_reg.item()
#
#         if i % 100 == 99:
#             print('Training - [%d, %5d] loss: %.3f' %
#                   (epoch + 1, i + 1, running_loss / 100))
#             running_loss = 0.0
#
#     # Validation
#     backNet.eval()
#     with torch.no_grad():
#
#         for i, data in enumerate(validloader):
#             inputs, keypoints = data['image'].to(device, dtype=torch.float), data['keypoints'].to(device)
#
#             reg_out = backNet(inputs)
#
#             loss_reg = criterion_reg(reg_out, keypoints)
#             valid_loss += loss_reg
#
#         print('Validation - [%d, %5d] loss: %.3f' %
#               (epoch + 1, i + 1, valid_loss / len(test_dataset)))
#         if min_valid > (valid_loss / len(test_dataset)):
#             min_valid = valid_loss / len(test_dataset)
#             torch.save(backNet.state_dict(), 'BackROI.pt')
#
#         print(min_valid)
#
#         valid_loss = 0.0
#
#         # plt.figure()
#         # plt.plot(1000, 250)
#         # show_landmarks_batch(data, reg_out)
#         # plt.axis('off')
#         # plt.ioff()
#         # plt.show()
#
#     print('-' * 250)
#
# print('Finished Training')

# for l in range(len(dataset)):
#
#     ID = dataset[l]['label']
#     ROI_coordinates = keypoints_to_coordinates(dataset[l]['points'], ROI_data=True)
#     image = io.imread(dataset[l]['image_path'])
#
#     with torch.no_grad():
#         image_copy = image.copy()
#         image = cv2.resize(image, (224, 224))
#         image = image.transpose((2, 0, 1))
#         image = np.expand_dims(image, 0)
#         image = torch.FloatTensor(image).cuda()
#         cls_out, reg_out = net(image)
#         reg_out = reg_out.cpu().numpy()
#         reg_out = keypoints_to_coordinates(reg_out)

# if reg_out[0][1] == 0.0 and reg_out[2][0] == 0.0:
#     continue
#
# h, w = image_copy.shape[0], image_copy.shape[1]
#
# for q in range(4):
#     reg_out[q][0] *= w / 224
#     reg_out[q][1] *= h / 224
#
# for q in range(4):
#     if reg_out[q][0] < 0.0:
#         reg_out[q][0] = 0.0
#     if reg_out[q][1] < 0.0:
#         reg_out[q][1] = 0.0
#
# affine_pts = np.array(reg_out, dtype=np.int)
# rect = cv2.boundingRect(affine_pts)
# x, y, w, h = rect
# cropped = image_copy[y:y + h, x:x + w].copy()
# dst = cv2.bitwise_and(cropped, cropped)
#
# for i in range(reg_out.shape[0]):
#     reg_out[i][0] -= x
#     reg_out[i][1] -= y
#
# for i in range(ROI_coordinates.shape[0]):
#     if ROI_coordinates[i][0] == 0.0 and ROI_coordinates[i][4] == 0.0:
#         continue
#     ROI_coordinates[i][0] -= x
#     ROI_coordinates[i][1] -= y
#     ROI_coordinates[i][2] -= x
#     ROI_coordinates[i][3] -= y
#     ROI_coordinates[i][4] -= x
#     ROI_coordinates[i][5] -= y
#     ROI_coordinates[i][6] -= x
#     ROI_coordinates[i][7] -= y
#
# for q in range(5):
#     if ROI_coordinates[i][0] < 0.0:
#         ROI_coordinates[i][0] = 0.0
#     if ROI_coordinates[i][1] < 0.0:
#         ROI_coordinates[i][1] = 0.0
#     if ROI_coordinates[i][2] < 0.0:
#         ROI_coordinates[i][2] = 0.0
#     if ROI_coordinates[i][3] < 0.0:
#         ROI_coordinates[i][3] = 0.0
#     if ROI_coordinates[i][4] < 0.0:
#         ROI_coordinates[i][4] = 0.0
#     if ROI_coordinates[i][5] < 0.0:
#         ROI_coordinates[i][5] = 0.0
#     if ROI_coordinates[i][6] < 0.0:
#         ROI_coordinates[i][6] = 0.0
#     if ROI_coordinates[i][7] < 0.0:
#         ROI_coordinates[i][7] = 0.0
#
# data = {}
# data['shapes'] = [{}, {}, {}, {}, {}]
# for i in range(5):
#     data['shapes'][i]['label'] = []
#     data['shapes'][i]['label'] = '{}-{}'.format(ID, i + 1)
#     data['shapes'][i]['points'] = []
#     data['shapes'][i]['points'] = ROI_coordinates[i].tolist()
#
# if ID == 'F':
#     with open('Front_ROI_Dataset/Training Data/Labels/{}.json'.format(front_names), 'w') as outfile:
#         json.dump(data, outfile)
#
#     cv2.imwrite('Front_ROI_Dataset/Training Data/{}.jpg'.format(front_names), cv2.cvtColor(dst, cv2.COLOR_RGB2BGR))
#
#     front_names += 1
#
# else:
#     with open('Back_ROI_Dataset/Training Data/Labels/{}.json'.format(back_names), 'w') as outfile:
#         json.dump(data, outfile)
#
#     cv2.imwrite('Back_ROI_Dataset/Training Data/{}.jpg'.format(back_names), cv2.cvtColor(dst, cv2.COLOR_RGB2BGR))
#
#     back_names += 1
#
# print("Number of images cropped: {}".format(k))
# k += 1

#             # print("Cropping completed!")
#
#             random_angle = np.random.randint(-30, 30)
#
#             # Crop
#
#             while True:
#                 key = random.randint(1, 50)
#                 background = cv2.imread("Background/{}.jpg".format(key))
#
#                 background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
#
#                 random_x = np.random.randint(0, int(background.shape[1] / 2))
#                 random_y = np.random.randint(0, int(background.shape[0] / 2))
#
#                 # Resize
#
#                 random_size = (np.random.random_sample() + 0.1) / 3 * background.shape[1]
#
#                 h, w = dst.shape[:2]
#                 r = random_size / float(h)
#                 dim = (int(w * r), int(random_size))
#                 dst = cv2.resize(dst, dim)
#
#                 if dst.shape[0] < background.shape[0] and dst.shape[1] < background.shape[1]:
#                     break
#
#             reg_out = reg_out.astype(float)
#
#             ROI_coordinates = ROI_coordinates.astype(float)
#
#             for i in range(4):
#                 reg_out[i][0] *= random_size / h
#                 reg_out[i][1] *= int(w * r) / w
#
#             for i in range(ROI_coordinates.shape[0]):
#                 if ROI_coordinates[i][0] == 0.0 and ROI_coordinates[i][4] == 0.0:
#                     continue
#                 ROI_coordinates[i][0] *= random_size / h
#                 ROI_coordinates[i][1] *= int(w * r) / w
#                 ROI_coordinates[i][2] *= random_size / h
#                 ROI_coordinates[i][3] *= int(w * r) / w
#                 ROI_coordinates[i][4] *= random_size / h
#                 ROI_coordinates[i][5] *= int(w * r) / w
#                 ROI_coordinates[i][6] *= random_size / h
#                 ROI_coordinates[i][7] *= int(w * r) / w
#
#             # print("Resizing completed!")
#
#             # Rotate
#
#             center_x_1 = dst.shape[1] / 2
#             center_y_1 = dst.shape[0] / 2
#
#             dst = rotate(dst, random_angle)
#
#             center_x_2 = dst.shape[1] / 2
#             center_y_2 = dst.shape[0] / 2
#
#             center_y_diff = center_y_2 - center_y_1
#             center_x_diff = center_x_2 - center_x_1
#
#             for i in range(4):
#                 reg_out[i][0] += center_x_diff
#                 reg_out[i][1] += center_y_diff
#
#             for i in range(ROI_coordinates.shape[0]):
#                 if ROI_coordinates[i][0] == 0.0 and ROI_coordinates[i][4] == 0.0:
#                     continue
#                 ROI_coordinates[i][0] += center_x_diff
#                 ROI_coordinates[i][1] += center_y_diff
#                 ROI_coordinates[i][2] += center_x_diff
#                 ROI_coordinates[i][3] += center_y_diff
#                 ROI_coordinates[i][4] += center_x_diff
#                 ROI_coordinates[i][5] += center_y_diff
#                 ROI_coordinates[i][6] += center_x_diff
#                 ROI_coordinates[i][7] += center_y_diff
#
#             reg_out[0] = rotate_coordinates(reg_out[0][0], reg_out[0][1], center_x_2, center_y_2, random_angle)
#             reg_out[1] = rotate_coordinates(reg_out[1][0], reg_out[1][1], center_x_2, center_y_2, random_angle)
#             reg_out[2] = rotate_coordinates(reg_out[2][0], reg_out[2][1], center_x_2, center_y_2, random_angle)
#             reg_out[3] = rotate_coordinates(reg_out[3][0], reg_out[3][1], center_x_2, center_y_2, random_angle)
#
#             for i in range(ROI_coordinates.shape[0]):
#                 if ROI_coordinates[i][0] == 0.0 and ROI_coordinates[i][4] == 0.0:
#                     continue
#                 ROI_coordinates[i][0], ROI_coordinates[i][1] = rotate_coordinates(ROI_coordinates[i][0],
#                                                                                   ROI_coordinates[i][1], center_x_2,
#                                                                                   center_y_2, random_angle)
#                 ROI_coordinates[i][2], ROI_coordinates[i][3] = rotate_coordinates(ROI_coordinates[i][2],
#                                                                                   ROI_coordinates[i][3], center_x_2,
#                                                                                   center_y_2, random_angle)
#                 ROI_coordinates[i][4], ROI_coordinates[i][5] = rotate_coordinates(ROI_coordinates[i][4],
#                                                                                   ROI_coordinates[i][5], center_x_2,
#                                                                                   center_y_2, random_angle)
#                 ROI_coordinates[i][6], ROI_coordinates[i][7] = rotate_coordinates(ROI_coordinates[i][6],
#                                                                                   ROI_coordinates[i][7], center_x_2,
#                                                                                   center_y_2, random_angle)
#
#             # print("Rotation completed!")
#
#             # Background
#
#             output, points, (translate_x, translate_y) = image_to_background(dst, background)
#
#             # Translate
#
#             for i in range(4):
#                 reg_out[i][0] += translate_y
#                 reg_out[i][1] += translate_x
#
#             for i in range(ROI_coordinates.shape[0]):
#                 if ROI_coordinates[i][0] == 0.0 and ROI_coordinates[i][4] == 0.0:
#                     continue
#                 ROI_coordinates[i][0] += translate_y
#                 ROI_coordinates[i][1] += translate_x
#                 ROI_coordinates[i][2] += translate_y
#                 ROI_coordinates[i][3] += translate_x
#                 ROI_coordinates[i][4] += translate_y
#                 ROI_coordinates[i][5] += translate_x
#                 ROI_coordinates[i][6] += translate_y
#                 ROI_coordinates[i][7] += translate_x
#
#             # print("Translation completed!")
#
#             data = {}
#             data['shapes'] = [{}, {}, {}, {}, {}]
#             for i in range(5):
#                 data['shapes'][i]['label'] = []
#                 data['shapes'][i]['label'] = '{}-{}'.format(ID, i + 1)
#                 data['shapes'][i]['points'] = []
#                 data['shapes'][i]['points'] = ROI_coordinates[i].tolist()
#
#             with open('ROI_Dataset_Original/Combined/Labels/{}.json'.format(k), 'w') as outfile:
#                 json.dump(data, outfile)
#
#             cv2.imwrite('ROI_Dataset_Original/Combined/{}.jpg'.format(k), cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
#
#             # print("Saving completed!")
#
#             print("Number of examples augmented: {}".format(k))
#             k += 1

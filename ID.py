from torch.utils.data import Dataset
from skimage import io
from utils import read_data

import torch
import torch.nn as nn
import torch.nn.functional as F


class IDCardsDataset(Dataset):
    """
    Dataset of ID card images.
    """

    def __init__(self, dataset_path, transform=None):
        self.data = read_data(dataset_path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        temp = self.data[idx]
        image = io.imread(temp['image_path'])
        sample = {'image': image, 'keypoints': temp['keypoints'], 'label': temp['label']}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """
    Class that defines a method for images of ID cards for converting them into Tensors.
    """

    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)
        return {'image': image,
                'keypoints': torch.FloatTensor(keypoints),
                'label': torch.FloatTensor(sample['label'])}


class Net(nn.Module):
    """
    CNN with ResNet152 backbone for finding keypoints of an ID card.
    """

    def __init__(self, resnet):
        super(Net, self).__init__()

        self.resnet = resnet

        self.fc1 = nn.Linear(100352, 120)
        self.fc2 = nn.Linear(120, 84)

        self.fc_cls = nn.Linear(84, 1)
        self.fc_reg = nn.Linear(84, 8)

    def forward(self, x):
        x = self.resnet(x)

        x = x.view(-1, self.num_flat_features(x))

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x_cls = self.fc_cls(x)
        x_cls = torch.sigmoid(x_cls)

        x_reg = self.fc_reg(x)

        return x_cls, x_reg

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


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

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
# train_dataset = IDCardsDataset("Resized_Dataset", transform=ToTensor())
# test_dataset = IDCardsDataset("Test_Dataset", transform=ToTensor())
#
# trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
# validloader = torch.utils.data.DataLoader(test_dataset, batch_size=4)
#
# print("Size of training set:", len(train_dataset))
# print("Size of testing set:", len(test_dataset))

# Step 2. Plotting a sample from the dataset

# image = cv2.imread("Resized_Dataset/Ablan Abkenov/1.jpg")
# with open("Resized_Dataset/Ablan Abkenov/Labels/1.json", 'r') as json_file:
#     json_data = json.load(json_file)
#     for p in json_data["shapes"]:
#         label = p["points"]
#
# show_image(image, label)
# print("STEP 2 - SUCCESS! Sample image was plotted!")

# Step 3. Preparing CNN, criterion and optimizer for training

# resnet = torchvision.models.resnet152(pretrained=True, progress=True).cuda()
# resnet.avgpool = Identity()
# resnet.fc = Identity()
#
# net = Net(resnet)
# net.to(device)
# net.load_state_dict(torch.load("IDregression.pt"))
#
# criterion_cls = nn.BCELoss()
# criterion_reg = nn.MSELoss()
# optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

# Step 4. Training loop

# model_check(net)
#
#     with open("Check/11.json", 'r') as json_file:
#         json_data = json.load(json_file)
#         for p in json_data["shapes"]:
#             label = p["points"]
#
#     reg_out = np.concatenate(reg_out)
#     keypoints = keypoints_to_coordinates(reg_out)
#     label = np.asarray(label)
#
#     image_copy = bird_view(image_copy, label, width=w)
#
#     config = "-l rus --oem 1 --psm 7"
#     print(image_to_string(image_copy, config=config))
#
#     show_image(image_copy, label)
#     # newline(p1, p2, color='red')
#     # newline(p2, p3, color='red')
#     # newline(p3, p4, color='red')
#     # newline(p4, p1, color='red')
#     # plt.show()

# net.eval()
# with torch.no_grad():
#
#     valid_loss = 0.0
#     accuracy = 0
#
#     for i, data in enumerate(validloader):
#         inputs, keypoints, labels = data['image'].to(device, dtype=torch.float), data['keypoints'].to(device), \
#                                     data['label'].to(device)
#         cls_out, reg_out = net(inputs)
#         loss_reg = criterion_reg(reg_out, keypoints)
#         valid_loss += loss_reg

# plt.figure()
# plt.plot(1000, 250)
# cls_out = cls_out >= 0.5
# labels = labels.type(torch.ByteTensor)
# labels = labels.to(device)
# show_landmarks_batch(data, reg_out)
# plt.axis('off')
# plt.ioff()
# plt.show()

# for i in range(len(cls_out)):
#     if cls_out[i] == labels[i]:
#         accuracy += 1

# print('-' * 250)

# print('Accuracy: %.3f%%' % (accuracy / len(test_dataset) * 100))
# print("Finished.")

# min_valid = 0.228
#
# for epoch in range(50):
#
#     net.train()
#     running_loss = 0.0
#     valid_loss = 0.0
#
#     print("Epoch: {}".format(epoch + 1))
#
#     for i, data in enumerate(trainloader):
#
#         inputs, keypoints, labels = data['image'].to(device, dtype=torch.float), data['keypoints'].to(device), data[
#             'label'].to(device)
#         optimizer.zero_grad()
#
#         ind = labels.squeeze(-1).type(torch.ByteTensor)
#
#         cls_out, reg_out = net(inputs)
#
#         loss_cls = criterion_cls(cls_out, labels)
#         loss_reg = criterion_reg(reg_out[ind], keypoints[ind])
#         total_loss = 0.7 * loss_cls + 0.3 * loss_reg
#         total_loss.backward()
#         optimizer.step()
#         running_loss += total_loss.item()
#
#         if i % 100 == 99:
#             print('Training - [%d, %5d] loss: %.3f' %
#                   (epoch + 1, i + 1, running_loss / 100))
#             running_loss = 0.0
#
#     # Validation
#     net.eval()
#     with torch.no_grad():
#
#         for i, data in enumerate(validloader):
#             inputs, keypoints, labels = data['image'].to(device, dtype=torch.float), data['keypoints'].to(device), \
#                                         data['label'].to(device)
#
#             ind = labels.squeeze(-1).type(torch.ByteTensor)
#
#             cls_out, reg_out = net(inputs)
#
#             loss_cls = criterion_cls(cls_out, labels)
#             loss_reg = criterion_reg(reg_out[ind], keypoints[ind])
#             total_loss = 0.5 * loss_cls + 0.5 * loss_reg
#             valid_loss += total_loss
#
#         print('Validation - [%d, %5d] loss: %.3f' %
#               (epoch + 1, i + 1, valid_loss / len(test_dataset)))
#         if min_valid > (valid_loss / len(test_dataset)):
#             min_valid = valid_loss / len(test_dataset)
#             torch.save(net.state_dict(), 'model_regcls.pt')
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

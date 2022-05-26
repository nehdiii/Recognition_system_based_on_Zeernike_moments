from ZernikeMomnets import ZernikeMoments
from Searcher import *
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
#import imutils
import json

import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim
from test import *
import numpy as np
from PIL import Image
import glob
from ZernikeMomnets import *
from DeepLearningMethodForSegmentation.data_loader import RescaleT
from DeepLearningMethodForSegmentation.data_loader import CenterCrop
from DeepLearningMethodForSegmentation.data_loader import ToTensor
from DeepLearningMethodForSegmentation.data_loader import ToTensorLab
from DeepLearningMethodForSegmentation.data_loader import SalObjDataset
from DeepLearningMethodForSegmentation.model import BASNet
import json




def normPRED(d):
	ma = torch.max(d)
	mi = torch.min(d)

	dn = (d-mi)/(ma-mi)

	return dn
model_dir = 'C:/Users/nehdi/PycharmProjects/PCD/DeepLearningMethodForSegmentation/saved_models/basnet_bsi/basnet.pth'


class ZernikeVectorGenerator:
    def __init__(self ,images_dir=None,test_dir=None):
        self.test_dir = test_dir
        self.images_dir = images_dir
        self.images_names= os.listdir(self.images_dir)
        self.data_base_json = "data_base_as_json_file"


    def GetImageName(self,image_name):
        return  image_name.replace(".png", "")

    def BasNetSegmentation(self,image_name,type=None):
        if type == None :
            iamge_to_sgment = [
                self.images_dir
                + "/"
                + image_name]
        if type == "inference":
            iamge_to_sgment = [
                self.test_dir
                + "/"
                + image_name]


        # --------- 2. dataloader ---------
        # 1. dataload
        test_salobj_dataset = SalObjDataset(img_name_list=iamge_to_sgment, lbl_name_list=[],
                                            transform=transforms.Compose([RescaleT(256), ToTensorLab(flag=0)]))
        test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1, shuffle=False, num_workers=1)

        # --------- 3. model define ---------
        net = BASNet(3, 1)
        net.load_state_dict(torch.load(model_dir))
        if torch.cuda.is_available():
            net.cuda()
        net.eval()

        # --------- 4. inference for each image ---------
        for i_test, data_test in enumerate(test_salobj_dataloader):
            inputs_test = data_test['image']
            inputs_test = inputs_test.type(torch.FloatTensor)

            if torch.cuda.is_available():
                inputs_test = Variable(inputs_test.cuda())
            else:
                inputs_test = Variable(inputs_test)

            d1, d2, d3, d4, d5, d6, d7, d8 = net(inputs_test)

            # normalization
            pred = d1[:, 0, :, :]
            pred = normPRED(pred)

            del d1, d2, d3, d4, d5, d6, d7, d8

            return pred

    def scale_and_translation_normlization(self,outline):
        beta = 800
        # calculate zeroyh order moment
        zeroth_order_raw_moments_m00 = cv2.moments(outline)['m00']
        # clalculate x_bar,y_bar for translation invariance and tranlate
        x_bar, y_bar = cv2.moments(outline)['m10'] / zeroth_order_raw_moments_m00, cv2.moments(outline)[
            'm01'] / zeroth_order_raw_moments_m00
        a = math.sqrt(beta/zeroth_order_raw_moments_m00)
        translation_matrix = np.float32([[a, 0, x_bar], [0, a, y_bar]])
        outline = cv2.warpAffine(outline, translation_matrix, (outline.shape[1], outline.shape[0]))
        return outline




    def test_GenerateZernikeVectors(self):
        data_base_dic = {}
        zer_calc = ZernikeMoments(21,21)
        for img in self.images_names:
            # get the image Name
            img_index = self.GetImageName(img)
            print(img)




            # forground back gound segmentation
            outline = self.BasNetSegmentation(img)
            outline = outline.detach().numpy()

            # correct the dim and out put for test
            outline = np.squeeze(outline,axis=0)
            outline = np.expand_dims(outline,axis=2)
            plt.imshow(outline ,cmap="gray")
            plt.show()

            # old method for zernike moment calculation the magnitude
            #moments = self.zer_calc.describe(np.squeeze(outline,axis=2))
            #moments = [moments[i]/zeroth_order_raw_moments_m00 for i in range(len(moments))]

           # translation invariance
            outline = self.scale_and_translation_normlization(outline)


            # claculate zernike moments
            #moments = zernike_moments(outline,21,21)

            moments = zer_calc.zernike_moments(outline)

            print(moments,img_index)

            data_base_dic[img_index] = moments
        return data_base_dic








    def test_predict(self,queryfeaturphat):


        zer_calc = ZernikeMoments(21,21)
        outline = self.BasNetSegmentation(queryfeaturphat,"inference")

        outline = outline.detach().numpy()
        outline = np.squeeze(outline, axis=0)
        outline = np.expand_dims(outline, axis=2)
        print(outline.shape)
        #plt.imshow(outline, cmap="gray")
        #plt.show()



        # translation and scale invariance
        outline = self.scale_and_translation_normlization(outline)

        # calculate zernike moments
        #test_moments = zernike_moments(outline,21,21)

        test_moments = zer_calc.zernike_moments(outline)

        print("test image")
        print(test_moments)


        vectorgenerator =self.test_GenerateZernikeVectors()

        #searcher = Searcher(vectorgenerator)
        searcher = SearcherOptimal(vectorgenerator)
        results = searcher.search(test_moments)
        return results

    def web_site_predictor(self,queryfeaturphat):
        outline = self.BasNetSegmentation(queryfeaturphat, "inference")
        outline = outline.detach().numpy()
        outline = np.squeeze(outline, axis=0)
        outline = np.expand_dims(outline, axis=2)
        plt.imshow(outline, cmap="gray")
        plt.show()

        outline = self.scale_and_translation_normlization(outline)


        test_moments = zernike_moments(outline, 21, 21)


        # load zernike data base vectors
        with open("static/DataBase/DataBaseImagesToZernikeVectors.json", "r") as openfile:
            vectorgenerator = json.load(openfile)


        # transfrom the data to adaptable format
        for var in vectorgenerator.keys():
            keys_values = vectorgenerator[var].items()
            new_d = {eval(key): complex(''.join(value.split())) for key, value in keys_values}
            vectorgenerator[var] = new_d


        # make prediction
        searcher = SearcherOptimal(vectorgenerator)
        results = searcher.search(test_moments)

        return [results[0][1],results[1][1],results[2][1],results[3][1]]

    def data_base_creator(self,filename):
        data = self.test_GenerateZernikeVectors()
        for var in data.keys():
            keys_values = data[var].items()
            new_d = {str(key): str(value) for key, value in keys_values}
            data[var] = new_d
        json_object = json.dumps(data, indent=4)
        with open("C:/Users/nehdi/PycharmProjects/PCD/static/DataBase/"+filename+".json", "w") as outfile:
            outfile.write(json_object)

    def API_Generate_Zernike_vectors_Data_Base(self,file_path,file_name):
        data = self.test_GenerateZernikeVectors()
        for var in data.keys():
            keys_values = data[var].items()
            new_d = {str(key): str(value) for key, value in keys_values}
            data[var] = new_d
        json_object = json.dumps(data, indent=4)
        with open( file_path + file_name + ".json", "w") as outfile:
            outfile.write(json_object)








if __name__ == "__main__":

    test = ZernikeVectorGenerator(
        'C:/Users/nehdi/PycharmProjects/PCD/static/bardo_pcd_pics',
        "C:/Users/nehdi/PycharmProjects/PCD/test_images_for_zernike"
    )
    print(test.web_site_predictor("img.png"))







import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

dir_path = os.path.dirname(os.path.abspath('POC_Flask/models'))

class Distortion_Classification(object):

    def __init__(self, img):
        self.img = img
        self.results = None
        self.msg = None

        try:
            ready, size = self.image_preprocess(self.img)
            if size == 256:
                model_256 = load_model(os.path.join(dir_path, 'models/Model_256.h5'))
                dist_pred = model_256.predict(ready)
                self.results = self.distortions(dist_pred)
            elif size == 128:
                model_128 = load_model(os.path.join(dir_path, 'models/Model_128.h5'))
                dist_pred = model_128.predict(ready)
                self.results = self.distortions(dist_pred)
            else:
                self.msg = '[ERROR] Image size unsupported'
                print(self.msg)

        except Exception as e:
            print('Error while classifying the image: ', e)


    ## Multiple distortions
    def distortions(self, preds):

        final_results = []
        try:
            sorted_probs = np.flip(np.sort(preds), axis=1)
            sorted_indeces = np.flip(np.argsort(preds), axis=1)
            threshold = 0.10
            distortions = []
            probs = []
            for i in range(3):
                if sorted_probs[0][i] >= threshold:
                    distortions.append(self.parse_label(sorted_indeces[0][i]))
                    probs.append(sorted_probs[0][i])

            ## Combine everything
            for i in range(len(distortions)):
                temp_dict = {}
                if i == 0:
                    temp_dict['Main Label'] = distortions[i][0]
                    temp_dict['Probability'] = probs[i]
                    temp_dict['Info'] = distortions[i][1]
                else:
                    temp_dict['Secundary Label'] = distortions[i][0]
                    temp_dict['Probability'] = probs[i]
                    temp_dict['Info'] = distortions[i][1]
                final_results.append(temp_dict)

            return final_results

        except Exception as e:
            print('Error while processing results: ', e)

    ## Image preprocessing function
    def image_preprocess(self, image):
        # Resize and Normalize
        try:
            if image.shape[0] > 64 and image.shape[1] > 64:
                if image.shape[0] < 1024 and image.shape[1] < 1024:
                    if image.shape[0] < 256 and image.shape[1] < 256:
                        resized = cv2.resize(image, (128, 128))
                        norm = np.asarray(resized).reshape(1, 128, 128, 3)
                        norm = norm / 255
                        size = 128
                        return norm, size
                    else:
                        resized = cv2.resize(image, (256, 256))
                        norm = np.asarray(resized).reshape(1, 256, 256, 3)
                        norm = norm / 255
                        size = 256
                        return norm, size
                else:
                    norm = 0
                    size = 1024
                    return norm, size

            else:
                norm = 0
                size = 64
                return norm, size

        except Exception as e:
            print('Error while preprocessing the image: ', e)

    ## Parse function
    def parse_label(self, label):
        try:
            if label == 0:
                label = 'Clean'
                extra_info = 'Image without distortions'
                return [label, extra_info]
            elif label == 1:
                label = 'Gaussian Blur Level 1'
                extra_info = 'Image with blur resembling a Gaussian distribution with std = [0.05, 2.5]'
                return [label, extra_info]
            elif label == 2:
                label = 'Gaussian Blur Level 2'
                extra_info = 'Image with blur resembling a Gaussian distribution with std = [4.5, 6.0]'
                return [label, extra_info]
            elif label == 3:
                label = 'Gaussian Blur Level 3'
                extra_info = 'Image with blur resembling a Gaussian distribution with std = [8.5, 10.0]'
                return [label, extra_info]
            elif label == 4:
                label = 'Gaussian Noise Level 1'
                extra_info = 'Image with noise resembling a Gaussian distribution with var = [0.005, 0.02]'
                return [label, extra_info]
            elif label == 5:
                label = 'Gaussian Noise Level 2'
                extra_info = 'Image with noise resembling a Gaussian distribution with var = [0.05, 0.065]'
                return [label, extra_info]
            elif label == 6:
                label = 'Gaussian Noise Level 3'
                extra_info = 'Image with noise resembling a Gaussian distribution with var = [0.10, 0.25]'
                return [label, extra_info]
            elif label == 7:
                label = 'High Brightness Level 1'
                extra_info = 'Image with brightness factor = [1.6, 1.9]'
                return [label, extra_info]
            elif label == 8:
                label = 'High Brightness Level 2'
                extra_info = 'Image with brightness factor = [2.7, 3.0]'
                return [label, extra_info]
            elif label == 9:
                label = 'JPEG Compression Level 1'
                extra_info = 'Image with artifacts related to JPEG compression with quality factors = [80, 35]'
                return [label, extra_info]
            elif label == 10:
                label = 'JPEG Compression Level 2'
                extra_info = 'Image with artifacts related to JPEG compression with quality factors = [20, 5]'
                return [label, extra_info]
            elif label == 11:
                label = 'Low Brightness Level 1'
                extra_info = 'Image with brightness factor = [0.8, 0.5]'
                return [label, extra_info]
            elif label == 12:
                label = 'Low Brightness Level 2'
                extra_info = 'Image with brightness factor = [0.3, 0.05]'
                return [label, extra_info]
            elif label == 13:
                label = 'Motion Blur Level 1'
                extra_info = 'Image with mild motion blur'
                return [label, extra_info]
            elif label == 14:
                label = 'Motion Blur Level 2'
                extra_info = 'Image with severe motion blur'
                return [label, extra_info]
            else:
                label = ''
                extra_info = 'No label to parse'
                return [label, extra_info]
        except Exception as e:
            print('Error while parsing labels: ', e)
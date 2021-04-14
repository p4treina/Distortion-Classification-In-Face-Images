# Distortion Classification Model for Face Images
#### Masters Dissertation Project
This repository provides two Convolutional Neural Network (CNN) models for distortion classification in face images. The two models are similar, differing only in that one was trained with images of size = (256,256), and the other with images of size = (128,128).
######
The models can classify undistorted images as well as images with Gaussian Noise, Gaussian Blur, Motion Blur, High Brightness, Low Brightness, and JPEG Compression. Three degradation levels were defined for the Gaussian blur and the Gaussian noise. For the other distortions two levels were defined, giving a total of 15 classes.

### **Classes:**
*   CL --> Clean/Undistorted images
*   GB1 --> Gaussian Blur level 1: Images with blur resembling a Gaussian distribution with std = [0.05, 2.5].
*   GB2 --> Gaussian Blur level 2: Images with blur resembling a Gaussian distribution with std = [4.5, 6.0].
*   GB3 --> Gaussian Blur level 3: Images with blur resembling a Gaussian distribution with std = [8.5, 10.0].
*   GN1 --> Gaussian Noise level 1: Images with noise resembling a Gaussian distribution with var = [0.005, 0.02].
*   GN2 --> Gaussian Noise level 2: Images with noise resembling a Gaussian distribution with var = [0.05, 0.065].
*   GN3 --> Gaussian Noise level 3: Images with noise resembling a Gaussian distribution with var = [0.10, 0.25].
*   HB1 --> High Brightness Level 1: Images with brightness factor = [1.6, 1.9].
*   HB2 --> High Brightness Level 2: Images with brightness factor = [2.5, 3.0].
*   JP1 --> JPEG Compression level 1: Images with artifacts related to JPEG compression with quality factors = [80, 35]. 
*   JP2 --> JPEG Compression level 2: Images with artifacts related to JPEG compression with quality factors = [20, 5].
*   MB1 --> Low Brightness level 1: Images with brightness factor = [0.8, 0.5].
*   MB2 --> Low Brightness level 2: Image with brightness factor = [0.3, 0.05].
*   LB1 --> Motion Blur level 1: Image with mild motion blur.
*   LB2 --> Motion Blur level 2: Image with severe motion blur.

### **Training:** (Distortion_Classification/train)
######
The models were trained in Google Colab with GPU and Tesorflow 2.4.1. We created our own dataset using a subset of images from https://github.com/NVlabs/ffhq-dataset as references to generate the distorted ones.
#### **Requirenments:**
*    tensorflow==2.4.1
*    opencv_python==4.1.2.30
*    numpy==1.19.5
*    livelossplot==0.5.3 (optional)
### **Test Results:**
*    Accuracy = 0.98
*    Precision = 0.98
*    Recall = 0.98
*    F1 Score = 0.98

### **Usage**:
######
example.ipynb

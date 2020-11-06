from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import cv2
import skimage.measure
import io, json
import base64
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
import joblib
from typing import List

app = FastAPI()

# define the Input class
class Input(BaseModel):
    brightness_r : float
    brightness_r_rsd : float
    color_temp_r : float
    planes : int
    image_base64 : str
    img_brightness_rsd : float
    pointcloud_x_string : str
    pointcloud_y_string : str
    pointcloud_z_string : str
    hologram_position_string : str

def base64str_to_OpenCVImage(image_base64):
    image = image_base64  # raw data with base64 encoding
    decoded_data = base64.b64decode(image)
    np_data = np.frombuffer(decoded_data,np.uint8)
    img_gray = cv2.imdecode(np_data,cv2.IMREAD_UNCHANGED)
    return img_gray

def camera_image_metrics(img_gray):
    img_brightness = (np.mean(img_gray)) / 255
    contrast = img_gray.std()
    entropy = skimage.measure.shannon_entropy(img_gray)
    sobel_h = cv2.Sobel(img_gray,cv2.CV_64F,1,0,ksize=3)
    sobel_v = cv2.Sobel(img_gray,cv2.CV_64F,0,1,ksize=3)
    SIr = np.sqrt(sobel_h**2 + sobel_v**2)
    SI = np.mean(SIr)
    orien = cv2.phase(np.array(sobel_h, np.float32), np.array(sobel_v, dtype=np.float32), angleInDegrees=True)
    GO_entropy = skimage.measure.shannon_entropy(orien)
    laplacian = cv2.Laplacian(img_gray, cv2.CV_64F).var()
    fast = cv2.FastFeatureDetector_create()
    kp = fast.detect(img_gray,None)
    corners = len(kp)
    
    return img_brightness, contrast, entropy, SI, GO_entropy, laplacian, corners

def pointcloud_metrics(pointcloud_x, pointcloud_y, pointcloud_z, hologram_position):
    x = np.array(pointcloud_x.split(","), dtype=float)
    y = np.array(pointcloud_y.split(","), dtype=float)
    z = np.array(pointcloud_z.split(","), dtype=float)
    fp_count = len(x)
    mean_depth = np.average(z)
    fp_density = fp_count / ((np.amax(x) - np.amin(x)) * (np.amax(y) - np.amin(y)) * (np.amax(z) - np.amin(z)))
    #Feature Point Proximity 
    position0 = np.array(hologram_position.split(","))
    distances = np.zeros(fp_count)
    for p in range(fp_count):
        dist_x = x[p] - float(position0[0])
        dist_y = y[p] - float(position0[1])
        dist_z = z[p] - float(position0[2])
        #Euclidean distance to hologram placement position
        distances[p] = np.sqrt((dist_x**2) + (dist_y**2) + (dist_z**2))
    fp_proximity = np.average(distances)
    #Spatial Heterogeneity
    x_r = robjects.FloatVector(x)
    y_r = robjects.FloatVector(y)
    z_r = robjects.FloatVector(z)
    
    robjects.r('''
        # create a function `spatial_het`
         spatial_het <- function(x,y,z) {
             library(spatstat)
             min_x <- min(x)
             max_x <- max(x)
             min_y <- min(y)
             max_y <- max(y)
             min_z <- min(z)
             max_z <- max(z)
             X <- pp3(x,z,y, box3(c(min_x,max_x),c(min_z,max_z),c(min_y,max_y)))
             K <- K3est(X)
             r <- K$r
             Lp <- ((K$theo)/pi)^(1/3)
             Lz_iso <- ((K$iso)/pi)^(1/3)
             sumDif <- 0
             for (t in 1:(length(Lp))) {
                     sumDif <- sumDif + abs(Lp[t] - Lz_iso[t])
             }
             print(sumDif)
        }
        ''')
        
    r_spatial_het = robjects.r['spatial_het']
    spatial_h = np.array(r_spatial_het(x_r,y_r,z_r))[0]

    return fp_count, mean_depth, fp_density, fp_proximity, spatial_h

class PredictResponse(BaseModel):
    data: List[float]

@app.put("/predict")
def predict(d:Input):
    #load model
    model = joblib.load('classifier.joblib')
    #platform data
    brightness_r = d.brightness_r
    brightness_r_rsd = d.brightness_r_rsd
    color_temp_r = d.color_temp_r
    planes = d.planes
	#image data
    img = base64str_to_OpenCVImage(d.image_base64)
    # Get image properties
    img_brightness, contrast, entropy, SI, GO_entropy, laplacian, corners = camera_image_metrics(img)
    img_brightness_rsd = d.img_brightness_rsd
    # Get pointcloud properties
    fp_count, mean_depth, fp_density, fp_proximity, spatial_h = pointcloud_metrics(d.pointcloud_x_string, d.pointcloud_y_string, d.pointcloud_z_string, d.hologram_position_string)
    
    X = np.array([brightness_r, brightness_r_rsd, planes, img_brightness, img_brightness_rsd, contrast, entropy, SI, GO_entropy, laplacian, corners, fp_count, mean_depth, fp_density, fp_proximity, spatial_h])
    X = X.reshape(1, -1)
    Y_predict = model.predict(X)
    result = PredictResponse(data=Y_predict.tolist())

    return result




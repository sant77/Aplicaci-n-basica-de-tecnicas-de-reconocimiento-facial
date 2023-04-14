# Features 
from skimage import feature
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import os
import glob
import cv2
import pathlib
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB

def extract_data(path, file_type="*.jpg",Resize=True):
    
    list_people = os.listdir(path)


    type_file = "*.jpg"

    cont_train = 0
    cont_train2 = 0

    for k in list_people:

        path_image = path + "/" + k
        list_jpg = glob.glob(path_image + os.sep + type_file)

        cont_train += len(list_jpg)


    print("El número de muestras para el entrenamiento: ",cont_train)

    #x,y = (cv2.imread(list_jpg[0], cv2.IMREAD_GRAYSCALE)).shape

    #train_data = np.zeros((cont_train,x,y))
    train_data = np.zeros((cont_train,192,168))
    train_data = train_data.astype('float32')

    y_train = np.zeros(cont_train)
    y_train = y_train.astype('int8')

    count = 0
    for i in list_people:

        #print(i)

        path_image = path + "/" + i

        list_jpg = glob.glob(path_image + os.sep + type_file)

        for j in list_jpg:


            image = cv2.imread(j, cv2.IMREAD_GRAYSCALE)
            if (Resize == True):
                #image = cv2.resize(image,(x,y))
                image = cv2.resize(image,(168,192))
                
            train_data[cont_train2] += image 

            y_train[cont_train2] = count
            #print(y_train[cont2])
            cont_train2 += 1
        count += 1
            
    return train_data, y_train



def blockshaped(arr, nrows, ncols):

    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
            .swapaxes(1,2)
            .reshape(-1, nrows, ncols))

def getHistogram(imgArray):
    hist, bin_edges = np.histogram(imgArray, density=True)
    return hist

def trainModel(model="knn"):

    path = str(pathlib.Path(__file__).parent.absolute())

    pathActual = path[0:len(path)-4]

    train_data, y_train = extract_data(pathActual +"/data")

    nsamples, nx, ny = train_data.shape

    train_data = train_data.reshape((nsamples,nx*ny))


    train_data = train_data.reshape((nsamples,nx,ny))


    train_data = train_data.astype("uint8")
    nsamples1, nx1, ny1 = train_data.shape
    train_data = train_data.reshape((nsamples1,nx1*ny1))
    train_data = train_data/255


    cont_train = train_data.shape[0]

    train_data_LBP = train_data.reshape((cont_train,nx,ny))
    lbp_feactures  = np.zeros((cont_train,nx,ny))

    eps = 1e-7
    radius = 1
    n_points = 8 * radius

    lbp_hist = np.zeros((cont_train,1440))

    for i in range(0,cont_train): 
        lbp = feature.local_binary_pattern(train_data_LBP[i], n_points, radius)
        lbp_feactures[i] +=  lbp
        
        shaped = blockshaped(lbp_feactures[i], 16, 14)
        x = []
        xBlocks = []
        for s in shaped:
            xBlocks.append(getHistogram(s))
        # Concatenate the various histogram, the resulting histogram is append into feature vector
        x.append(np.concatenate(xBlocks))
        
        lbp_hist[i] += x[0]

    #print(lbp_feactures)

    if model == "knn":
        clf_lbp_Kn_3 = KNeighborsClassifier(n_neighbors=3)
        clf_lbp_Kn_3 = clf_lbp_Kn_3.fit(lbp_hist,y_train)

        return clf_lbp_Kn_3
    
    elif model == "svm":
        param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
        'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }

        clf_lbp_svm = GridSearchCV(
            SVC(kernel='rbf', class_weight='balanced'), param_grid)
        
        clf_lbp_svm = clf_lbp_svm.fit(lbp_hist,y_train )

        return clf_lbp_svm

    elif model == "gauss":
        clf_lbp_Gauss = GaussianNB()
        clf_lbp_Gauss = clf_lbp_Gauss.fit(lbp_hist,y_train)

        return clf_lbp_Gauss
    else:
        return "Modelo incorrecto seleccionado"

def lbpImage(img):

    radius = 1
    n_points = 8 * radius

    lbp_hist = np.zeros((1,1440))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(gray,(168,192))

    lbp = feature.local_binary_pattern(image, n_points, radius)
    
    
    shaped = blockshaped(lbp, 16, 14)
    x = []
    xBlocks = []
    for s in shaped:
        xBlocks.append(getHistogram(s))
    # Concatenate the various histogram, the resulting histogram is append into feature vector
    x.append(np.concatenate(xBlocks))
    
    lbp_hist[0] += x[0]

    return lbp_hist



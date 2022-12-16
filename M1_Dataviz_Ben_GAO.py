# packages for preprocessing and visualization
import numpy as np
from os import listdir
import rasterio
import cv2
from matplotlib import pyplot as plt

# packages for pca and kernel pca
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA

# packages for auto-encoder and convolutionnal auto-encoder
from tensorflow import keras
from sklearn.model_selection import train_test_split

# package for wavelet transform
import pywt
''' 
    The code below loads, preprocesses, and normalizes dataset for PCA methods, this process is very long
    We have done it with our computer and save the results in imgs_469_431.npy, 
    so that one can just load the prepared dataset from .npy file

# function load images and resize them
def load_images(folder,nb_img,x,y) :
    loaded_images = list()
    nb = 0
    for filename in listdir(folder):
        # load image
        with rasterio.open(folder+'/' + filename, 'r') as img:
                array = img.read(1)  # read raster value (1 for one channel)
        # resize the image
        array = cv2.resize(array, dsize=(y, x), interpolation=cv2.INTER_AREA)
        # store loaded image
        loaded_images.append(array) 
        nb += 1
        if nb == nb_img:
            break
    print("finish download "+folder)
    return loaded_images

# selected folders (find explanation in report)
folders = ["F","G","H","I","J","M"]

# load and preprocess images
ds = list()
for folder in folders :
    # download 1000 images per class
    ds += load_images(folder,200,469,431)
ds = np.array(ds)

# show an image
plt.imshow(ds[500],cmap="gray")

# normalization
X = ds.reshape((ds.shape[0], ds.shape[1]*ds.shape[2])) #flattening the image 
# if you standardize a bigger dataset, the change to float32 is good for RAM
# X = X.astype('float32')    
X = StandardScaler().fit_transform(X) # standardize X

# printing a sample image to show the effect of standardization
plt.imshow(X[500].reshape(469,431), cmap='gray')

# save the nparray in imgs_469_431.npy
with open('imgs_469_431.npy', 'wb') as f:
    np.save(f, X)
'''




# the images are not of same size
# they vary for the first dimension between 469 and about 500, for second dimention from 431 to about 600
# so we chose to resize them to the lowest dimension of the dataset (469,431) to keep the most information as we can

# load 1200 standardized samples of size 202139=469*431 (200 per class)
with open('imgs_469_431.npy', 'rb') as f:
    X = np.load(f)
 
# creat the PCA model with 20 principal axis
pca_20 = PCA(n_components=20) 
# fit the model with X 
pca_20.fit(X) 
#pourcentage of total variance explained by each axis
print(pca_20.explained_variance_ratio_)
#pourcentage of information kept with principle axis
print(pca_20.explained_variance_ratio_.sum())
#eigen values
print(pca_20.singular_values_**2)	

#Apply the fitted PCA model to X and store the principal components in X_reduced
X_reduced=pca_20.transform(X)

# eigne images
eigen_imgs = pca_20.components_.reshape(20,469,431)
plt.imshow(eigen_imgs[0],cmap="gray")


# reconstruction of image
# temp = np.matmul(X_reduced[500],pca_20.components_)
# plt.imshow(temp.reshape(469,431),cmap="gray")

# prepare lables for visualization
labels = list()
for l in range(6):
    labels+=[l for i in range(200)]
print(labels)

# visualization with 2 principal axis
def pca_visu_2d() :
    plt.figure(figsize=(14,7))#instanciate an empty figure
    plt.scatter(X_reduced[:,0],X_reduced[:,1],c=labels)#plot a 2d point cloud
    plt.xlabel('pc1')#add a label to x-axis
    plt.ylabel('pc2')#add a label to y-axis
    plt.axvline(x=0,color="black")#add a vertical line
    plt.axhline(y=0,color="black")#add a horizontal line
    plt.show()

pca_visu_2d()

# visualization with 3 principal axis
def pca_visu_3d() :
    fig = plt.figure(figsize=(14,7))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X_reduced[:,0],X_reduced[:,1], X_reduced[:,2], marker="o",c = labels)
    ax.set_xlabel('pc1')
    ax.set_ylabel('pc2')
    ax.set_zlabel('pc3')
    plt.show()
    
pca_visu_3d()




# kernel PCA

transformer = KernelPCA(n_components=7, kernel='rbf',gamma = 1/(469*431))
X_transformed = transformer.fit_transform(X)
print(transformer.eigenvalues_)
# visualization with 2 principal axis
def kernel_pca_visu_2d() :
    plt.figure(figsize=(14,7))#instanciate an empty figure
    plt.scatter(X_transformed[:,0],X_transformed[:,1],c=labels)#plot a 2d point cloud
    plt.xlabel('pc1')#add a label to x-axis
    plt.ylabel('pc2')#add a label to y-axis
    plt.axvline(x=0,color="black")#add a vertical line
    plt.axhline(y=0,color="black")#add a horizontal line
    plt.show()

kernel_pca_visu_2d()

# visualization with 3 principal axis
def kernel_pca_visu_3d() :
    fig = plt.figure(figsize=(14,7))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X_transformed[:,0],X_transformed[:,1], X_transformed[:,2], marker="o",c = labels)
    ax.set_xlabel('pc1')
    ax.set_ylabel('pc2')
    ax.set_zlabel('pc3')
    plt.show()
    
kernel_pca_visu_3d()





'''
    The code below loads, preprocesses, and normalizes dataset for auto-encoder methods, this process is very long
    We have done it with our computer and save the results in imgs_128_128.npy, 
    so that one can just load the prepared dataset from .npy file

ds2 = list()
for folder in ["M"] :
    # download all images with 9999 (the largest class contains less than 5000 images)
    ds2 += load_images(folder,9999,128,128)
ds2 = np.array(ds2)  
  
# show an image
plt.imshow(ds2[500],cmap="gray")

# min max normalisation
X2 = ds2/ds2.max()

# show a sample image to show the effect of normalization
# plt.imshow(X2[500], cmap='gray')

# save X2
with open('imgs_128_128.npy', 'wb') as f:
    np.save(f, X2)
'''
   

# load images resized to 128*128 and normalized by min-max normalization
# the choice of resizing images to 128*128 is due to the fact we have 25000 images
# the resolution 469*431=202139 will destoy all efforts to train the network, 128*128=16348 is a compromise
with open('imgs_128_128.npy', 'rb') as f:
    X2 = np.load(f)

Y2 = [ 1 for i in range(0,4900) ]+[ 2 for i in range(4900,9697) ]+[ 3 for i in range(9697,14295) ] + [ 4 for i in range(14295,19035) ]+[ 5 for i in range(19035,23744) ]+[ 6 for i in range(23744,25904) ]
# train set and validation set
X2_train, X2_valid, Y2_train, Y2_test = train_test_split(X2, Y2, test_size=0.20, random_state=42)


# convolutionnal autoencoder 
# construct the network
# (the batch-normalizarion is chronophage)
conv_encoder = keras.models.Sequential([
    keras.layers.Reshape([128, 128, 1], input_shape=[128,128]),
    keras.layers.Conv2D(8, kernel_size=3, padding="same", activation="relu", kernel_initializer="he_normal"),
    keras.layers.MaxPool2D(pool_size=2),
    # keras.layers.BatchNormalization(),
    keras.layers.Conv2D(16, kernel_size=3, padding="same", activation="relu", kernel_initializer="he_normal"),
    keras.layers.MaxPool2D(pool_size=2),
    #keras.layers.BatchNormalization(),
    keras.layers.Conv2D(32, kernel_size=3, padding="same", activation="relu", kernel_initializer="he_normal"),
    keras.layers.MaxPool2D(pool_size=2),
    #keras.layers.BatchNormalization(),
    keras.layers.Conv2D(64, kernel_size=3, padding="same", activation="relu", kernel_initializer="he_normal"),
    keras.layers.MaxPool2D(pool_size=2),
    keras.layers.Flatten(input_shape=[8, 8, 64]),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(3, activation=None)
])
conv_decoder = keras.models.Sequential([
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(8*8*64, activation="relu"),
    keras.layers.Reshape([8, 8, 64], input_shape=[8*8*64]),
    #keras.layers.BatchNormalization(),
    keras.layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding="same", activation="relu", kernel_initializer="he_normal"),
    #keras.layers.BatchNormalization(),
    keras.layers.Conv2DTranspose(16, kernel_size=3, strides=2, padding="same", activation="relu", kernel_initializer="he_normal"),
    #keras.layers.BatchNormalization(),
    keras.layers.Conv2DTranspose(8, kernel_size=3, strides=2, padding="same", activation="relu", kernel_initializer="he_normal"),
    #keras.layers.BatchNormalization(),
    keras.layers.Conv2DTranspose(1, kernel_size=3, strides=2, padding="same", activation="sigmoid"),
    keras.layers.Reshape([128, 128])
])
conv_ae = keras.models.Sequential([conv_encoder, conv_decoder])
conv_ae.compile(loss="binary_crossentropy",optimizer=keras.optimizers.Adam(learning_rate=0.001))


# train the network
history_conv = conv_ae.fit(X2_train, X2_train, epochs=10, validation_data=[X2_valid, X2_valid], batch_size=32)
#conv_ae.evaluate(X2_valid,X2_valid)


# show the loss
'''
def loss_cae():
    plt.plot(history_conv.history["loss"], label="train_loss")
    plt.plot(history_conv.history["val_loss"], label="valid_loss")
    plt.legend(loc="upper right")
loss_cae()
'''

# encode the images
conv_codings = conv_encoder.predict(X2_train)

# reconstruction
def plot_image(image):
    plt.imshow(image, cmap="binary")
    plt.axis("off")
def show_reconstructions(model, n_images=5):
    reconstructions = model.predict(X2_valid[:n_images])
    fig = plt.figure(figsize=(n_images * 1.5, 3))
    for image_index in range(n_images):
        plt.subplot(2, n_images, 1 + image_index)
        plot_image(X2_valid[image_index])
        plt.subplot(2, n_images, 1 + n_images + image_index)
        plot_image(reconstructions[image_index])
   
#show_reconstructions(ae)
show_reconstructions(conv_ae)

# prepare lables for visualization
labels = Y2_train[0:1200]

# visualization with 2 principal axis
def cae_visu_2d() :
    plt.figure(figsize=(14,7))#instanciate an empty figure
    plt.scatter(conv_codings[0:1200,0],conv_codings[0:1200,1],c=labels)#plot a 2d point cloud
    plt.xlabel('dimension 1 du vecteur latent')#add a label to x-axis
    plt.ylabel('dimension 2 du vecteur latent')#add a label to y-axis
    plt.show()

cae_visu_2d()
# visualization with 3 principal axis
def cae_visu_3d() :
    fig = plt.figure(figsize=(14,7))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(conv_codings[0:1200,2],conv_codings[0:1200,0], conv_codings[0:1200,1],c = labels)
    ax.set_xlabel('dimension 3 du vecteur latent')
    ax.set_ylabel('dimension 1 du vecteur latent')
    ax.set_zlabel('dimension 2 du vecteur latent')
    plt.show()
    
cae_visu_3d()


# Using Discret Wavelet Transform to viusalize individualy an image
img_wave = X[500].reshape(469,431)

# wavelet transform
coeffs2=pywt.dwt2(img_wave,'db3',mode='periodization')

# cA is the Approximation coefficients
# cH is the Horizental detailed coefficients
# cV is the Ventical detailed coefficients
# cD is the Diaganal detailed coefficients
cA, (cH, cV, cD) = coeffs2

plt.figure(figsize=(20,20))

plt.subplot(2,2,1)
plt.imshow(cA,cmap='gray')
plt.title('cA: Approximation Coeff.', fontsize=30)

plt.subplot(2,2,2)
plt.imshow(cH,cmap='gray')
plt.title('cH: Horizontal Detailed Coeff.', fontsize=30)

plt.subplot(2,2,3)
plt.imshow(cV,cmap='gray')
plt.title('cV: Vertical Detailed Coeff.', fontsize=30)

plt.subplot(2,2,4)
plt.imshow(cD,cmap='gray')
plt.title('cD: Diagonal Detailed Coeff.', fontsize=30)

plt.show()
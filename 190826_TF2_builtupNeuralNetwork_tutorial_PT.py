import os
import numpy as np
from tensorflow import keras
from pyrsgis import raster
from pyrsgis.convert import changeDimension
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score

#影像储存文件夹！！！！！！
os.chdir("C:/Users/Ma Dawei/Desktop/Classification _of_satellite_remote_sensing_images/Landsat_5_TM_images")

#Assign file names
mxBangalore = 'l5_Bangalore2011_raw.tif'  #训练影像
builtupBangalore = 'l5_Bangalore2011_builtup.tif'#训练影像对应的标记数据
mxHyderabad = 'l5_Hyderabad2011_raw.tif' #测试数据

# 使用pyrsgis库中的raster模块同时读取遥感影像的地理坐标以及DN值，并且存放于numpy数组中
ds1, featuresBangalore = raster.read(mxBangalore, bands='all')
ds2, labelBangalore = raster.read(builtupBangalore, bands=1)
ds3, featuresHyderabad = raster.read(mxHyderabad, bands='all')

# Print the size of the arrays
print("Bangalore Multispectral image shape: ", featuresBangalore.shape)
print("Bangalore Binary built-up image shape: ", labelBangalore.shape)
print("Hyderabad Multispectral image shape: ", featuresHyderabad.shape)


# Clean the labelled data to replace NoData values by zero
labelBangalore = (labelBangalore == 1).astype(int)

#print("Bangalore Multispectral image shape: ", featuresBangalore)

print("Bangalore Multispectral image shape: ", labelBangalore[0])


# Reshape the array to single dimensional array
featuresBangalore = changeDimension(featuresBangalore)
labelBangalore = changeDimension (labelBangalore)
featuresHyderabad = changeDimension(featuresHyderabad)
nBands = featuresBangalore.shape[1]

print("Bangalore Multispectral image shape: ", featuresBangalore.shape)
print("Bangalore Binary built-up image shape: ", labelBangalore.shape)
print("Hyderabad Multispectral image shape: ", featuresHyderabad.shape)

#Split testing and training datasets
xTrain, xTest, yTrain, yTest = train_test_split(featuresBangalore, labelBangalore, test_size=0.4, random_state=42)

print(xTrain.shape)
print(yTrain.shape)

print(xTest.shape)
print(yTest.shape)

# Normalise the data
xTrain = xTrain / 255.0
xTest = xTest / 255.0
featuresHyderabad = featuresHyderabad / 255.0

# Reshape the data
xTrain = xTrain.reshape((xTrain.shape[0], 1, xTrain.shape[1]))
xTest = xTest.reshape((xTest.shape[0], 1, xTest.shape[1]))
featuresHyderabad = featuresHyderabad.reshape((featuresHyderabad.shape[0], 1, featuresHyderabad.shape[1]))

# Print the shape of reshaped data
print(xTrain.shape, xTest.shape, featuresHyderabad.shape)

# Define the parameters of the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(1, nBands)),
    keras.layers.Dense(14, activation='relu'),
    keras.layers.Dense(2, activation='softmax')])

# Define the accuracy metrics and parameters
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Run the model
model.fit(xTrain, yTrain, epochs=1)

# Predict for test data 
yTestPredicted = model.predict(xTest)
yTestPredicted = yTestPredicted[:,1]

# Calculate and display the error metrics
yTestPredicted = (yTestPredicted>0.5).astype(int)
cMatrix = confusion_matrix(yTest, yTestPredicted)
pScore = precision_score(yTest, yTestPredicted)
rScore = recall_score(yTest, yTestPredicted)

print("Confusion matrix: for 14 nodes\n", cMatrix)
print("\nP-Score: %.3f, R-Score: %.3f" % (pScore, rScore))

predicted = model.predict(featuresHyderabad)
predicted = predicted[:,1]

# Predict new data and export the probability raster
prediction = np.reshape(predicted, (ds3.RasterYSize, ds3.RasterXSize))
outFile = 'Hyderabad_2011_BuiltupNN_predicted_5.tif'
raster.export(prediction, ds3, filename=outFile, dtype='float')

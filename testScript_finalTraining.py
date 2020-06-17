import largeScale_net as network
import imageio
import numpy as np
import os
from pathlib import Path
import cv2
import shutil 
inputSize = 512
strideSize = 256
exampleSize = 256
def gaussianWeight(x, mu, sigma):
    a = 1/(sigma * np.sqrt(2.0 * np.pi))
    return a * np.exp(-(1.0/2.0) * ((x - mu)/sigma)**2)

#This function crops an image in smaller inputSize tiles, with a stride strideSize and saves it in the given output_dir/test
def cropImage(imagePath, materialName, output_dir):
    output_test_dir = os.path.join(output_dir, "test")
    if not os.path.exists(output_test_dir):
        os.makedirs(output_test_dir)

    currentImageName = materialName
    image = imageio.imread(imgPath)
    height = int(image.shape[0])
    width = int(image.shape[1])

    widthSplit = int(np.ceil(width / strideSize)) - 1
    heightSplit = int(np.ceil(height / strideSize)) - 1

    #Split the image
    maxIDImage = 0

    for i in range(widthSplit):
        for j in range(heightSplit):
            currentJPix = j * strideSize
            currentIPix = i * strideSize
            split = image[currentJPix:currentJPix + inputSize, currentIPix:currentIPix + inputSize, :]
            splitID = (i * heightSplit) + j

            currentSplitPath = os.path.join(output_test_dir, currentImageName + "_" + str(splitID) + ".png")

            im = imageio.imwrite(currentSplitPath, split)

            maxIDImage = splitID
    return maxIDImage, widthSplit, heightSplit, height, width

#Takes an exemplar image (with normal, diffuse roughness and specular maps concatenated along X axis) in the form of a numpy array and creates a resized version of the exemplar in a "train" folder as these will be used for the fine tuning.
def resizeCropAndCreateTestSet(crops, materialName, output_dir):
    output_train_dir = os.path.join(output_dir, "train")
    
    if not os.path.exists(output_train_dir):
        os.makedirs(output_train_dir)

    cropIm = imageio.imread(crops)
    cropsNb = int(cropIm.shape[0] // exampleSize)
    crops =  np.split(cropIm, cropsNb, axis = 0)
    
    for id, crop in enumerate(crops):
        splitted = np.split(crop,4, axis = 1)
        resized = []
        for image in splitted:
            resized.append(cv2.resize(image, (588,588), interpolation = cv2.INTER_LANCZOS4))
            
        result = np.concatenate(resized, axis = 1)
        imageio.imwrite(os.path.join(output_train_dir, materialName + "_example" + str(id) + "_resized.png"), result)
        
    return cropsNb
    
def stitchResults(inputSize, outputFolder, maxIDImage, networkOutputsFolder, materialName, height, width, widthSplit, heightSplit):
    #We define here the weight to apply to each tiles, with one in the center and 0 in the borders.
    sigm = 0.20
    maxVal = gaussianWeight(0.5, 0.5, sigm)
    oneImageWeights = np.asarray(np.meshgrid(np.linspace(0, 1, inputSize), np.linspace(0, 1, inputSize)))
    oneImageWeights = gaussianWeight(oneImageWeights, 0.5, sigm) / maxVal
    oneImageWeights = oneImageWeights[0] * oneImageWeights[1]
    oneImageWeights = np.expand_dims(oneImageWeights, axis=-1)

    #Which folder is going to hold the stitched final results.
    folderOutput = os.path.join(outputFolder, "results_fineTuned")
    if not os.path.exists(folderOutput):
        os.makedirs(folderOutput)

    #for each map id (representing normal, diffuse, roughness and specular)
    for idMap in range(4):
        allImages = []
        #We store in memory all the results for the different tiles for the current map type.
        for idImage in range(maxIDImage + 1):
            imagePath = os.path.join(networkOutputsFolder,"final", "images", materialName + "_" + str(idImage)+"-outputs_gammad-" + str(idMap) + "-.png" )
            allImages.append(imageio.imread(imagePath))

        #We initialize the final image and the weights we will use to normalize the contribution of each tile
        finalImage = np.zeros((height, width, 3))
        finalWeights = np.zeros((height, width, 3))
        for i in range(widthSplit):
            for j in range(heightSplit):
                currentJPix = j * strideSize
                currentIPix = i * strideSize
                splitID = (i * heightSplit) + j
                #We now paste each images weight by the gaussian weights in the final image at the proper position
                finalImage[currentJPix:currentJPix + inputSize, currentIPix:currentIPix + inputSize, :] = finalImage[currentJPix:currentJPix + inputSize, currentIPix:currentIPix + inputSize, :] + ((allImages[splitID]/255.0) * oneImageWeights)
                #And creates a final weight image that stores the different total weights applied to each pixel, to normalize it in the end
                finalWeights[currentJPix:currentJPix + inputSize, currentIPix:currentIPix + inputSize, :] = finalWeights[currentJPix:currentJPix + inputSize, currentIPix:currentIPix + inputSize, :] +oneImageWeights
        #Normalizes the image with respect to each pixel's weight.
        finalImage = finalImage / finalWeights
        #Saves the map as uint8.
        finalImage = (finalImage * 255).astype(np.uint8)
        imageio.imwrite(os.path.join(folderOutput, materialName + "_" + str(idMap) + ".png"), finalImage)

materialNames = []
input_dir = "dataExample/"
#Extract all the materials to be processed. This assumes that the data are stored with naming "name".png for the large scale image and "name"_example.png for the exemplars (each exemplar concatenated along the Y axis).
for dirpath, dirnames, filenames in os.walk(input_dir):
    for filename in filenames:
        if "_example" in filename and not filename.split("_example")[0] in materialNames:
            materialNames.append(filename.split("_example")[0])
    break
    
print(materialNames)

#We use the checkpoint of the pre-trained network.
checkpoint = "saved_weights/"

outputFolder = os.path.join(input_dir, "output")
if not os.path.exists(outputFolder):
    os.makedirs(outputFolder)
    
#For each of the material for which we have an exemplar, we process it.     
for matNb, materialName in enumerate(materialNames):
    crops = os.path.join(input_dir, materialName + "_example.png")
    if not os.path.exists(crops):
        continue #If we can't find the exemplar, move on to the next material.
    imgPath = os.path.join(input_dir, materialName + ".png")
    if not os.path.exists(imgPath):
        imgPath = os.path.join(input_dir, materialName + ".jpg") #If the png doesn't exist, check if it could be a jpg
    
    if not os.path.exists(imgPath):
        print("/!\/!\/!\ Could not find an image, skipping : " + str(imgPath))
        continue #If the large scale image cannot be found, move on to the next material
        
    #Defines the folder that will be created to store the images needed for the fine tuning.
    inputFolder = os.path.join(outputFolder, "postTraining", materialName)
    if not os.path.exists(inputFolder):
        os.makedirs(inputFolder)

    #Here we prepare the tiles split from the large scale guide image.
    maxIDImage, widthSplit, heightSplit, height, width = cropImage(imgPath, materialName, inputFolder)
    
    #Here we prepare the exemplar to be fine tuned on.
    resizeCropAndCreateTestSet(crops, materialName, inputFolder)
    
    print(materialName)
    print(str(matNb + 1) + "/"+str(len(materialNames)))
    #Read imageFile
    #Run on the folder with all images still providing the same material cropImage
    #get all the results

    #Define the output dir of the network runs
    networkOutputs = os.path.join(outputFolder, "networkOutputs")
    
    #Run the network as a training, using the checkpoint for the pre trained network. The nbStepMax is the number of steps of fine tuning, 1000 seems to be more than enough.
    network.runNetwork(inputFolder, networkOutputs, checkpoint, inputMode = "folder", feedMethod = "render", mode="finetune", input_size=588, nbTargets = 4, batch_size = 1, nbStepMax = 1000, testApproach="files")
    
    stitchResults(inputSize, outputFolder, maxIDImage, networkOutputs, materialName, height, width, widthSplit, heightSplit)
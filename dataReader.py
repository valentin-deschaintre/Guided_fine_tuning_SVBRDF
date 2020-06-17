import os
import glob
import tensorflow as tf
from random import shuffle
import helpers
import math
import renderer
import multiprocessing
import numpy as np

class dataset:
    inputPath = ""
    trainFolder = "train"
    testFolder = "test"
    pathList  = []
    imageFormat = "png"
    which_direction = "AtoB"
    nbTargetsToRead = 4

    logInput = False
    fixCrop = False
    mixMaterials = True
    useAmbientLight = False
    useAugmentationInRenderings = True

    tileSize = 256
    inputImageSize = 288

    batchSize = 1
    iterator = None
    inputBatch = None
    targetBatch = None
    pathBatch = None
    gammaCorrectedInputsBatch = None
    stepsPerEpoch = 0
    
    #Some default constructor with most important parameters
    def __init__(self, inputPath, trainFolder = "train", testFolder = "test", nbTargetsToRead = 4, tileSize=256, inputImageSize=288, batchSize=1, imageFormat = "png", which_direction = "AtoB", fixCrop = False, mixMaterials = True, logInput = False, useAmbientLight = False, useAugmentationInRenderings = True):
        self.inputPath = inputPath
        self.trainFolder = trainFolder
        self.testFolder = testFolder
        self.nbTargetsToRead = nbTargetsToRead
        self.tileSize = tileSize
        self.inputImageSize = inputImageSize
        self.batchSize = batchSize
        self.imageFormat = imageFormat
        self.fixCrop = fixCrop
        self.mixMaterials = mixMaterials
        self.logInput = logInput
        self.useAmbientLight = useAmbientLight
        self.useAugmentationInRenderings = useAugmentationInRenderings

    #Public function to populate the list of path for this dataset
    def loadPathList(self, inputMode, runMode, randomizeOrder, inputpythonList):
        if self.inputPath is None or not os.path.exists(self.inputPath):
            raise ValueError("The input path doesn't exist :(!")

        if inputMode == "folder":
            self.__loadFromDirectory(runMode, randomizeOrder)
        if inputMode == "image":
            self.pathList = [self.inputPath]
        if inputMode == "pythonList":
            self.pathList = inputpythonList

    #Handles the reading of files from a directory
    def __loadFromDirectory(self, runMode, randomizeOrder = True):
        modeFolder = ""
        if runMode == "train" or runMode == "finetune":
            modeFolder = self.trainFolder
        elif runMode == "test":
            modeFolder = self.testFolder

        path = os.path.join(self.inputPath, modeFolder)
        fileList = []
        for rootDir, dirs, filenames in os.walk(path):
            for filename in filenames:
                if filename.endswith(self.imageFormat):
                    fileList.append(os.path.join(rootDir, filename))
        fileList = sorted(fileList)
        if randomizeOrder:
            shuffle(fileList)

        if not fileList:
            raise ValueError("The list of filepaths is empty :( : " + path)
        self.pathList = fileList
        
    #Handles the reading of a material + examples
    def __readImagesGT(self, filename):
        material_string = tf.read_file(filename) #Gets a string tensor from a file
        decodedInput = tf.image.decode_image(material_string) #Decode a string tensor as image
        floatMaterial = tf.image.convert_image_dtype(decodedInput, dtype=tf.float32) #Transform image to float32
        assertion = tf.assert_equal(tf.shape(floatMaterial)[-1], 3, message="image does not have 3 channels")

        with tf.control_dependencies([assertion]):
            #floatInput = tf.identity(floatInput)#Not sure why the use of a identity node is required here (related to the control dependency apparently). Maybe will need to readdit
            floatMaterial.set_shape([None, None, 3])
            
        targets = tf.stack(tf.split(floatMaterial, self.nbTargetsToRead, axis=1, name="Split_input_data1"), axis = 0) 
            
        return filename, targets
        
    #Handles the reading of a single image
    def __readImages(self,filename):
        image_string = tf.read_file(filename) #Gets a string tensor from a file
        decodedInput = tf.image.decode_image(image_string) #Decode a string tensor as image
        floatInput = tf.image.convert_image_dtype(decodedInput, dtype=tf.float32) #Transform image to float32
        assertion = tf.assert_equal(tf.shape(floatInput)[-1], 3, message="image does not have 3 channels")

        with tf.control_dependencies([assertion]):
            floatInput.set_shape([None, None, 3])
        gammadInput = floatInput
        #print("CAREFUL THE GAMMA IS NOT CORRECTED AUTOMATICALLY")
        #input = floatInput
        input = tf.pow(floatInput, 2.2) #correct for the gamma
        #If we want to log the inputs, we do it here
        if self.logInput:
            input = helpers.logTensor(input)
        #The preprocess function puts the vectors value between [-1; 1] from [0;1]
        input = helpers.preprocess(input)

        targets = tf.zeros(tf.shape(input)) # is here (None, None, 3)
        targets = tf.expand_dims(targets, axis = 0)
        targets = tf.tile(targets, (self.nbTargetsToRead, 1,1,1))

        return filename, input, targets, gammadInput


    def __readMaterial(self, material):
        image_string = tf.read_file(material) #Gets a string tensor from a file
        decodedInput = tf.image.decode_image(image_string) #Decode a string tensor as image
        floatMaterial = tf.image.convert_image_dtype(decodedInput, dtype=tf.float32) #Transform image to float32

        return material, tf.split(floatMaterial, self.nbTargetsToRead, axis=1, name="Split_input_data1")

    def __renderInputs(self, materials, renderingScene, jitterLightPos, jitterViewPos, mixMaterials):
        fullSizeMixedMaterial = materials
        if mixMaterials:
            alpha = tf.random_uniform([1], minval=0.1, maxval=0.9, dtype=tf.float32, name="mixAlpha")

            materials1 = materials[::2]
            materials2 = materials[1::2]

            fullSizeMixedMaterial = helpers.mixMaterials(materials1, materials2, alpha)

        if self.inputImageSize >=  self.tileSize :
            if self.fixCrop:
                xyCropping = (self.inputImageSize - self.tileSize) // 2
                xyCropping = [xyCropping, xyCropping]
            else:
                xyCropping = tf.random_uniform([2], 0, self.inputImageSize - self.tileSize, dtype=tf.int32)
            cropped_mixedMaterial = fullSizeMixedMaterial[:,:, xyCropping[0] : xyCropping[0] + self.tileSize, xyCropping[1] : xyCropping[1] + self.tileSize, :]
        elif self.inputImageSize < self.tileSize:
            raise Exception("Size of the input is inferior to the size of the rendering, please provide higher resolution maps")
        cropped_mixedMaterial.set_shape([None, self.nbTargetsToRead, self.tileSize, self.tileSize, 3])
        mixedMaterial = helpers.adaptRougness(cropped_mixedMaterial)

        targetstoRender = helpers.target_reshape(mixedMaterial) #reshape it to be compatible with the rendering algorithm [?, size, size, 12]
        nbRenderings = 1
        rendererInstance = renderer.GGXRenderer(includeDiffuse = True)
        ## Do renderings of the mixedMaterial

        targetstoRender = helpers.preprocess(targetstoRender) #Put targets to -1; 1
        surfaceArray = helpers.generateSurfaceArray(self.tileSize)

        inputs = helpers.generateInputRenderings(rendererInstance, targetstoRender, self.batchSize, nbRenderings, surfaceArray, renderingScene, jitterLightPos, jitterViewPos, self.useAmbientLight, useAugmentationInRenderings = self.useAugmentationInRenderings)

        self.gammaCorrectedInputsBatch =  tf.squeeze(inputs, [1])

        inputs = tf.pow(inputs, 2.2) # correct gamma
        if self.logInput:
            inputs = helpers.logTensor(inputs)

        inputs = helpers.preprocess(inputs) #Put inputs to -1; 1

        targets = helpers.target_deshape(targetstoRender, self.nbTargetsToRead)
        return targets, inputs

    def populateInNetworkFeedGraph(self, renderingScene, jitterLightPos, jitterViewPos, shuffle = True):
        #Create a tensor out of the list of paths
        filenamesTensor = tf.constant(self.pathList)
        #Reads a slice of the tensor, for example, if the tensor is of shape [100,2], the slice shape should be [2]
        dataset = tf.data.Dataset.from_tensor_slices(filenamesTensor)
        #for each slice apply the __readImages function
        dataset = dataset.map(self.__readMaterial, num_parallel_calls= int(multiprocessing.cpu_count() / 4)) #Divided by four as the cluster divides cpu availiability for each GPU
        #Authorize repetition of the dataset when one epoch is over.
        dataset = dataset.repeat()
        if shuffle:
            dataset = dataset.shuffle(buffer_size=64, reshuffle_each_iteration=True)
        #set batch size
        #print(self.batchSize)
        nbWithdraw = 2 * self.batchSize
        if not self.mixMaterials:
            nbWithdraw = self.batchSize
        batched_dataset = dataset.batch(nbWithdraw)
        batched_dataset = batched_dataset.prefetch(buffer_size=2)
        #batched_dataset = batched_dataset.cache()

        iterator = batched_dataset.make_initializable_iterator()

        #Create the node to retrieve next batch
        init_paths_batch, init_targets_batch = iterator.get_next()

        paths_batch = init_paths_batch

        if self.mixMaterials:
            paths_batch = init_paths_batch[::2]

        targets_batch, inputs_batch = self.__renderInputs(init_targets_batch, renderingScene, jitterLightPos, jitterViewPos, self.mixMaterials)

        inputs_batch = tf.squeeze(inputs_batch, [1])#Before this the input has a useless dimension in 1 as we have only 1 rendering
        inputs_batch.set_shape([None, self.tileSize, self.tileSize, 3])
        targets_batch.set_shape([None, self.nbTargetsToRead, self.tileSize, self.tileSize, 3])
        print("steps per epoch: " + str(int(math.floor(len(self.pathList) / nbWithdraw))))
        #Populate the object
        self.stepsPerEpoch = int(math.floor(len(self.pathList) / nbWithdraw ))
        self.inputBatch = inputs_batch
        self.targetBatch = targets_batch
        self.iterator = iterator
        self.pathBatch = paths_batch

    #This function if used to create the iterator to go over the data and create the tensors of input and output
    def populateFeedGraph(self, shuffle = True, cutPaperOut = False):
        with tf.name_scope("load_images"):
            #Create a tensor out of the list of paths
            filenamesTensor = tf.constant(self.pathList)
            #Reads a slice of the tensor, for example, if the tensor is of shape [100,2], the slice shape should be [2] (to check if we have problem here)
            dataset = tf.data.Dataset.from_tensor_slices(filenamesTensor)

            #for each slice apply the __readImages function
            dataset = dataset.map(self.__readImages, num_parallel_calls=int(multiprocessing.cpu_count() / 4))
            #Authorize repetition of the dataset when one epoch is over.
            if shuffle:
               dataset = dataset.shuffle(buffer_size=16, reshuffle_each_iteration=True)
            #set batch size
            dataset = dataset.repeat()
            batched_dataset = dataset.batch(self.batchSize)
            batched_dataset = batched_dataset.prefetch(buffer_size=4)
            #Create an iterator to be initialized
            iterator = batched_dataset.make_initializable_iterator()

            #Create the node to retrieve next batch
            paths_batch, inputs_batch, targets_batch, gammadInputBatch = iterator.get_next()

            self.gammaCorrectedInputsBatch = gammadInputBatch
            reshaped_targets = helpers.target_reshape(targets_batch)
            #inputRealSize = self.tileSize
            inputRealSize = self.inputImageSize

            #Do the random crop, if the crop if fix, crop in the middle
            if inputRealSize > self.tileSize:
                if self.fixCrop:
                    xyCropping = (inputRealSize - self.tileSize) // 2
                    xyCropping = [xyCropping, xyCropping]
                else:
                    xyCropping = tf.random_uniform([1], 0, inputRealSize - self.tileSize, dtype=tf.int32)

                inputs_batch = inputs_batch[:,:, xyCropping[0] : xyCropping[0] + self.tileSize, xyCropping[0] : xyCropping[0] + self.tileSize, :]
                targets_batch = targets_batch[:,:, xyCropping[0] : xyCropping[0] + self.tileSize, xyCropping[0] : xyCropping[0] + self.tileSize, :]

            #Set shapes
            inputs_batch.set_shape([None, self.tileSize, self.tileSize, 3])
            targets_batch.set_shape([None, self.nbTargetsToRead, self.tileSize, self.tileSize, 3])

            #Populate the object
            self.stepsPerEpoch = int(math.floor(len(self.pathList) / self.batchSize))
            self.inputBatch = inputs_batch
            self.targetBatch = targets_batch
            self.iterator = iterator
            self.pathBatch = paths_batch

    #This function if used to crate the iterator to go over the data and create the tensors of input and output
    def populateInNetworkFeedGraphSpatialMix(self,renderingScene, shuffle = True, imageSize = 512, useSpatialMix = True):
        with tf.name_scope("load_images"):
            #Create a tensor out of the list of paths
            filenamesTensor = tf.constant(self.pathList)
            #Reads a slice of the tensor, for example, if the tensor is of shape [100,2], the slice shape should be [2] (to check if we have problem here)
            dataset = tf.data.Dataset.from_tensor_slices(filenamesTensor)

            #for each slice apply the __readImages function
            dataset = dataset.map(self.__readImagesGT, num_parallel_calls=int(multiprocessing.cpu_count() / 4))
            #Authorize repetition of the dataset when one epoch is over.
            #shuffle = True
            if shuffle:
               dataset = dataset.shuffle(buffer_size=16, reshuffle_each_iteration=True)
            #set batch size
            dataset = dataset.repeat()
            toPull = self.batchSize 
            if useSpatialMix:
                toPull = self.batchSize * 2
            batched_dataset = dataset.batch(toPull)
            batched_dataset = batched_dataset.prefetch(buffer_size=4)
            #Create an iterator to be initialized
            iterator = batched_dataset.make_initializable_iterator()

            #Create the node to retrieve next batch
            paths_batch, targets_batch = iterator.get_next()
            inputRealSize = imageSize#Should be input image size but changed tmp
            
            if useSpatialMix:
                threshold = 0.5
                perlinNoise = tf.expand_dims(tf.expand_dims(helpers.generate_perlin_noise_2d((inputRealSize, inputRealSize), (1,1)), axis = -1), axis = 0)
                perlinNoise = (perlinNoise + 1.0) * 0.5
                perlinNoise = perlinNoise >= threshold
                perlinNoise = tf.cast(perlinNoise, tf.float32)
                inverted = 1.0 - perlinNoise

                materialsMixed1 = targets_batch[::2] * perlinNoise
                materialsMixed2 = targets_batch[1::2] * inverted
                
                fullSizeMixedMaterial = materialsMixed1 + materialsMixed2
                targets_batch = fullSizeMixedMaterial
                paths_batch = paths_batch[::2]                
                
            targetstoRender = helpers.target_reshape(targets_batch) #reshape it to be compatible with the rendering algorithm [?, size, size, 12]
            nbRenderings = 1
            rendererInstance = renderer.GGXRenderer(includeDiffuse = True)
            ## Do renderings of the mixedMaterial
            mixedMaterial = helpers.adaptRougness(targetstoRender)

            targetstoRender = helpers.preprocess(targetstoRender) #Put targets to -1; 1
            surfaceArray = helpers.generateSurfaceArray(inputRealSize)

            inputs_batch = helpers.generateInputRenderings(rendererInstance, targetstoRender, self.batchSize, nbRenderings, surfaceArray, renderingScene, False, False, self.useAmbientLight, useAugmentationInRenderings = self.useAugmentationInRenderings)
            
            targets_batch = helpers.target_deshape(targetstoRender, self.nbTargetsToRead)
            self.gammaCorrectedInputsBatch =  tf.squeeze(inputs_batch, [1])
            #tf.summary.image("GammadInputs", helpers.convert(inputs[0, :]), max_outputs=5)
            inputs_batch = tf.pow(inputs_batch, 2.2) # correct gamma
            if self.logInput:
                inputs_batch = helpers.logTensor(inputs_batch)

            #Do the random crop, if the crop if fix, crop in the middle
            if inputRealSize > self.tileSize:
                if self.fixCrop:
                    xyCropping = (inputRealSize - self.tileSize) // 2
                    xyCropping = [xyCropping, xyCropping]
                else:
                    xyCropping = tf.random_uniform([1], 0, inputRealSize - self.tileSize, dtype=tf.int32)

                inputs_batch = inputs_batch[:, :, xyCropping[0] : xyCropping[0] + self.tileSize, xyCropping[0] : xyCropping[0] + self.tileSize, :]
                targets_batch = targets_batch[:,:, xyCropping[0] : xyCropping[0] + self.tileSize, xyCropping[0] : xyCropping[0] + self.tileSize, :]

            #Set shapes
            inputs_batch = tf.squeeze(inputs_batch, [1]) #Before this the input has a useless dimension in 1 as we have only 1 rendering
            inputs_batch.set_shape([None, self.tileSize, self.tileSize, 3])
            targets_batch.set_shape([None, self.nbTargetsToRead, self.tileSize, self.tileSize, 3])
            
            #Populate the object
            self.stepsPerEpoch = int(math.floor(len(self.pathList) / self.batchSize))
            self.inputBatch = inputs_batch
            self.targetBatch = targets_batch
            self.iterator = iterator
            self.pathBatch = paths_batch

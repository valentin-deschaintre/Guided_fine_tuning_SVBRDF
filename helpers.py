import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import math
import acquisitionScene
#import renderer

def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1

def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2

def logTensor(tensor):
    return  (tf.log(tf.add(tensor,0.01)) - tf.log(0.01)) / (tf.log(1.01)-tf.log(0.01))

def tf_generate_normalized_random_direction(batchSize, nbRenderings, lowEps = 0.001, highEps = 0.05):
    r1 = tf.random_uniform([batchSize, nbRenderings, 1], 0.0 + lowEps, 1.0 - highEps, dtype=tf.float32)
    r2 =  tf.random_uniform([batchSize, nbRenderings, 1], 0.0, 1.0, dtype=tf.float32)
    r = tf.sqrt(r1)
    phi = 2 * math.pi * r2
    #min alpha = atan(sqrt(1-r^2)/r)
    x = r * tf.cos(phi)
    y = r * tf.sin(phi)
    z = tf.sqrt(1.0 - tf.square(r))
    finalVec = tf.concat([x, y, z], axis=-1) #Dimension here should be [batchSize,nbRenderings, 3]
    return finalVec

def removeGamma(tensor):
    return tf.pow(tensor, 2.2)

def addGamma(tensor):
    return tf.pow(tensor, 0.4545)

def target_reshape(targetBatch):
    #Here the target batch is [?(Batchsize), 4, 256, 256, 3] and we want to go to [?(Batchsize), 256,256,12]
    return tf.concat(tf.unstack(targetBatch, axis = 1), axis = -1)

def target_deshape(target, nbTargets):
    #target have shape [batchsize, 256,256,12] and we want [batchSize,4, 256,256,3]
    target_list = tf.split(target, nbTargets, axis=-1)#4 * [batch, 256,256,3]
    return tf.stack(target_list, axis = 1) #[batch, 4,256,256,3]

def tf_generate_distance(batchSize, nbRenderings):
    gaussian = tf.random_normal([batchSize, nbRenderings, 1], 0.5, 0.75, dtype=tf.float32) # parameters chosen empirically to have a nice distance from a -1;1 surface.
    return (tf.exp(gaussian))


def NormalizeIntensity(tensor):
    maxValue = tf.reduce_max(tensor)
    return tensor / maxValue

# Normalizes a tensor troughout the Channels dimension (BatchSize, Width, Height, Channels)
# Keeps 4th dimension to 1. Output will be (BatchSize, Width, Height, 1).
def tf_Normalize(tensor):
    Length = tf.sqrt(tf.reduce_sum(tf.square(tensor), axis = -1, keepdims=True))
    return tf.math.divide(tensor, Length)

# Computes the dot product between 2 tensors (BatchSize, Width, Height, Channels)
# Keeps 4th dimension to 1. Output will be (BatchSize, Width, Height, 1).
def tf_DotProduct(tensorA, tensorB):
    return tf.reduce_sum(tf.multiply(tensorA, tensorB), axis = -1, keepdims=True)

def tf_lampAttenuation(distance):
    DISTANCE_ATTENUATION_MULT = 0.001
    return 1.0 / (1.0 + DISTANCE_ATTENUATION_MULT*tf.square(distance))

def tf_lampAttenuation_pbr(distance):
    return 1.0 / tf.square(distance)

def squeezeValues(tensor, min, max):
    return tf.clip_by_value(tensor, min, max)

#This allows to avoid having completely mirror objects
def adaptRougness(mixedMaterial):
    #Material has shape [Batch, 4, 256,256,3]
    multiplier = [1.0, 1.0, 0.9, 1.0]
    addition = [0.0, 0.0, 0.1, 0.0]
    multiplier = tf.reshape(multiplier, [1,4,1,1,1])
    addition = tf.reshape(addition, [1,4,1,1,1])
    return (mixedMaterial * multiplier) + addition

def mixMaterials(material1, material2, alpha):
    normal1Corrected = (material1[:, 0] - 0.5) * 2.0 #go between -1 and 1
    normal2Corrected = (material2[:, 0] - 0.5) * 2.0
    normal1Projected = normal1Corrected / tf.expand_dims(tf.maximum(0.01, normal1Corrected[:,:,:,2]), axis = -1) #Project the normals to use the X and Y derivative
    normal2Projected = normal2Corrected / tf.expand_dims(tf.maximum(0.01, normal2Corrected[:,:,:,2]), axis = -1)

    mixedNormals = alpha * normal1Projected  + (1.0 - alpha) * normal2Projected
    normalizedNormals = mixedNormals / tf.sqrt(tf.reduce_sum(tf.square(mixedNormals), axis=-1, keepdims=True))
    normals = (normalizedNormals * 0.5) + 0.5 # Back to 0;1

    mixedRest = alpha * material1[:, 1:] + (1.0 - alpha) * material2[:, 1:]

    final = tf.concat([tf.expand_dims(normals, axis = 1), mixedRest], axis = 1)

    return final

def generateSurfaceArray(crop_size, pixelsToAdd = 0):
    totalSize = crop_size + (pixelsToAdd * 2)
    surfaceArray=[]
    XsurfaceArray = tf.expand_dims(tf.lin_space(-1.0, 1.0, totalSize), axis=0)
    XsurfaceArray = tf.tile(XsurfaceArray,[totalSize, 1])
    YsurfaceArray = -1 * tf.transpose(XsurfaceArray) #put -1 in the bottom of the table
    XsurfaceArray = tf.expand_dims(XsurfaceArray, axis = -1)
    YsurfaceArray = tf.expand_dims(YsurfaceArray, axis = -1)

    surfaceArray = tf.concat([XsurfaceArray, YsurfaceArray, tf.zeros([totalSize, totalSize,1], dtype=tf.float32)], axis=-1)
    surfaceArray = tf.expand_dims(tf.expand_dims(surfaceArray, axis = 0), axis = 0)#Add dimension to support batch size and nbRenderings
    return surfaceArray

#found here: https://github.com/pvigier/perlin-numpy
def generate_perlin_noise_2d(shape, res):
    def f(t):
        return 6*t**5 - 15*t**4 + 10*t**3
    
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    
    grid = np.transpose(np.meshgrid(np.arange(0, res[0], delta[0]), np.arange(0, res[1], delta[1])), (1, 2, 0)) % 1
    grid = grid.astype(np.float32)
    grid = grid * 3.0
    grid = tf.constant(grid)
    # Gradients
    angles = 2 * np.pi * tf.random.uniform((res[0]+1, res[1]+1), dtype = tf.float32)
    gradients = tf.stack((tf.math.cos(angles), tf.math.sin(angles)), axis = -1)
    g00 = tf.tile(gradients[0:-1,0:-1],(d[0], d[1], 1))
    g10 = tf.tile(gradients[1:,0:-1],(d[0], d[1], 1))
    g01 = tf.tile(gradients[0:-1,1:],(d[0], d[1], 1))
    g11 = tf.tile(gradients[1:,1:],(d[0], d[1], 1))
    # Ramps
    n00 = tf.reduce_sum(grid * g00, 2)
    n10 = tf.reduce_sum(tf.stack((grid[:,:,0]-1, grid[:,:,1]), axis = -1) * g10, -1)
    n01 = tf.reduce_sum(tf.stack((grid[:,:,0], grid[:,:,1]-1), axis = -1) * g01, -1)
    n11 = tf.reduce_sum(tf.stack((grid[:,:,0]-1, grid[:,:,1]-1), axis = -1) * g11, -1)
    # Interpolation
    t = f(grid)
    n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
    n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
    return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)


def jitterPosAround(batchSize, nbRenderings, posTensor, mean = 0.0, stddev = 0.03):
    randomPerturbation =  tf.clip_by_value(tf.random_normal([batchSize, nbRenderings,1,1,1,3], mean, stddev, dtype=tf.float32), -0.05, 0.05) #Clip here how far it can go to 8 * stddev to avoid negative values on view or light ( Z minimum value is 0.3)
    return posTensor + randomPerturbation

#"staticViewPlaneLight", "staticViewHemiLight", "staticViewHemiLightOneSurface", "movingViewHemiLight", "movingViewHemiLightOneSurface"
def generateInputRenderings(rendererInstance, material, batchSize, nbRenderings, surfaceArray, renderingScene, jitterLightPos, jitterViewPos, useAmbientLight, useAugmentationInRenderings = True, nbVariationOfRender = 1):
    currentLightPos, currentViewPos, currentConeTargetPos = None, None, None
    if renderingScene == "staticViewPlaneLight":
        currentLightPos, currentViewPos, currentConeTargetPos, surfaceArray, useAugmentation = acquisitionScene.defaultScene(surfaceArray, batchSize, nbRenderings)
    elif renderingScene == "staticViewSpotLight":
        currentLightPos, currentViewPos, currentConeTargetPos, surfaceArray, useAugmentation = acquisitionScene.defaultSceneSpotLight(surfaceArray, batchSize, nbRenderings)
    elif renderingScene == "staticViewHemiSpotLight":
        currentLightPos, currentViewPos, currentConeTargetPos, surfaceArray, useAugmentation = acquisitionScene.fixedViewHemisphereConeLight(surfaceArray, batchSize, nbRenderings)
    elif renderingScene == "staticViewHemiSpotLightOneSurface":
        currentLightPos, currentViewPos, currentConeTargetPos, surfaceArray, useAugmentation = acquisitionScene.fixedViewHemisphereConeLightOneOnPlane(surfaceArray, batchSize, nbRenderings)
    elif renderingScene == "movingViewHemiSpotLightOneSurface":
        currentLightPos, currentViewPos, currentConeTargetPos, surfaceArray, useAugmentation = acquisitionScene.movingViewHemisphereConeLightOneOnPlane(surfaceArray, batchSize, nbRenderings, useAugmentationInRenderings)
    elif renderingScene == "diffuse":
        currentLightPos, currentViewPos, currentConeTargetPos, surfaceArray, useAugmentation = acquisitionScene.diffuse(surfaceArray, batchSize, nbRenderings, useAugmentationInRenderings)
    elif renderingScene == "moreSpecular":
        currentLightPos, currentViewPos, currentConeTargetPos, surfaceArray, useAugmentation = acquisitionScene.moreSpecular(surfaceArray, batchSize, nbRenderings, useAugmentationInRenderings)        
    elif renderingScene == "collocatedHemiSpotLightOneSurface":
        currentLightPos, currentViewPos, currentConeTargetPos, surfaceArray, useAugmentation = acquisitionScene.movingViewHemisphereConeLightOneOnPlane(surfaceArray, batchSize, nbRenderings, useAugmentationInRenderings)
        currentViewPos = currentLightPos
    elif renderingScene == "fixedAngles":
        currentLightPos, currentViewPos, currentConeTargetPos, surfaceArray, useAugmentation = acquisitionScene.fixedAngles(surfaceArray, batchSize, nbRenderings, useAugmentationInRenderings)
    else:
        raise ValueError("Rendering scene unknown")

    if not useAugmentationInRenderings:
        useAugmentation = False

    print("use Augmentation ? : " + str(useAugmentation))
    if jitterLightPos and not currentLightPos is None:
        currentLightPos = jitterPosAround(batchSize, nbRenderings, currentLightPos, 0.0, 0.01)
    if jitterViewPos and not currentViewPos is None and renderingScene != "collocatedHemiSpotLightOneSurface":
        currentViewPos = jitterPosAround(batchSize, nbRenderings, currentViewPos, 0.0, 0.03)

    wo = currentViewPos - surfaceArray
    if useAmbientLight:
        ambientLightPos = acquisitionScene.generateAmbientLight(currentLightPos, batchSize)
        wiAmbient = ambientLightPos - surfaceArray
        renderingAmbient = rendererInstance.tf_Render(material, wiAmbient, wo, None, multiLight = True, currentLightPos = ambientLightPos, lossRendering = False, isAmbient = True, useAugmentation = useAugmentation)[0]
    wi = currentLightPos - surfaceArray

    renderings = rendererInstance.tf_Render(material,wi,wo, currentConeTargetPos, multiLight = True, currentLightPos = currentLightPos, lossRendering = False, isAmbient = False, useAugmentation = useAugmentation)[0] 
    if useAmbientLight:
        renderings = renderings + renderingAmbient #Add ambient if necessary
        renderingAmbient = tf.clip_by_value(renderingAmbient, 0.0, 1.0)
        renderingAmbient = tf.pow(renderingAmbient, 0.4545)
        renderingAmbient = tf.image.convert_image_dtype(convert(renderingAmbient), dtype=tf.float32)
        tf.summary.image("renderingAmbient", convert(renderingAmbient[0, :]), max_outputs=5)
    if useAugmentation:
        renderings = addNoise(renderings)
    renderings = tf.clip_by_value(renderings, 0.0, 1.0) # Make sure noise doesn't put values below 0 and simulate over exposure
    renderings = tf.pow(renderings, 0.4545) #gamma the results

    renderings = tf.image.convert_image_dtype(convert(renderings), dtype=tf.float32)
    return renderings

def addNoise(renderings):
    shape = tf.shape(renderings)
    stddevNoise = tf.exp(tf.random_normal((), mean = np.log(0.005), stddev=0.3))
    noise = tf.random_normal(shape, mean=0.0, stddev=stddevNoise)
    return renderings + noise

def tf_generateDiffuseRendering(batchSize, nbRenderings, targets, outputs, renderer):
    currentViewPos = tf_generate_normalized_random_direction(batchSize, nbRenderings, lowEps = 0.001, highEps = 0.1)
    currentLightPos = tf_generate_normalized_random_direction(batchSize, nbRenderings, lowEps = 0.001, highEps = 0.1)

    wi = currentLightPos
    wi = tf.expand_dims(wi, axis=2)
    wi = tf.expand_dims(wi, axis=2)

    wo = currentViewPos
    wo = tf.expand_dims(wo, axis=2)
    wo = tf.expand_dims(wo, axis=2)

    #Here we have wi and wo with shape [batchSize, height,width, nbRenderings, 3]
    renderedDiffuse = renderer.tf_Render(targets,wi,wo, None, "diffuse", useAugmentation = False, lossRendering = True)[0]

    renderedDiffuseOutputs = renderer.tf_Render(outputs,wi,wo, None, "", useAugmentation = False, lossRendering = True)[0]#tf_Render_Optis(outputs,wi,wo)
    return [renderedDiffuse, renderedDiffuseOutputs]


def tf_generateSpecularRendering(batchSize, nbRenderings, surfaceArray, targets, outputs, renderer):
    currentViewDir = tf_generate_normalized_random_direction(batchSize, nbRenderings, lowEps = 0.001, highEps = 0.1)
    currentLightDir = currentViewDir * tf.expand_dims([-1.0, -1.0, 1.0], axis = 0)
    #Shift position to have highlight elsewhere than in the center.
    currentShift = tf.concat([tf.random_uniform([batchSize, nbRenderings, 2], -1.0, 1.0), tf.zeros([batchSize, nbRenderings, 1], dtype=tf.float32) + 0.0001], axis=-1)

    currentViewPos = tf.multiply(currentViewDir, tf_generate_distance(batchSize, nbRenderings)) + currentShift
    currentLightPos = tf.multiply(currentLightDir, tf_generate_distance(batchSize, nbRenderings)) + currentShift

    currentViewPos = tf.expand_dims(currentViewPos, axis=2)
    currentViewPos = tf.expand_dims(currentViewPos, axis=2)

    currentLightPos = tf.expand_dims(currentLightPos, axis=2)
    currentLightPos = tf.expand_dims(currentLightPos, axis=2)

    wo = currentViewPos - surfaceArray
    wi = currentLightPos - surfaceArray

    renderedSpecular = renderer.tf_Render(targets,wi,wo, None, "specu", useAugmentation = False, lossRendering = True)[0]
    renderedSpecularOutputs = renderer.tf_Render(outputs,wi,wo, None, "", useAugmentation = False, lossRendering = True)[0]

    return [renderedSpecular, renderedSpecularOutputs]

def optimistic_saver(save_file):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables() if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    name2var = dict(zip(map(lambda x:x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = name2var[saved_var_name]
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
    return tf.train.Saver(restore_vars)

def should(freq, max_steps, step):
    return freq > 0 and ((step + 1) % freq == 0)

def process_targets(targets):
    diffuse = targets[:,:,:,3:6]
    normals = targets[:,:,:,0:3]
    roughness = targets[:,:,:,6:9]
    specular = targets[:,:,:,9:12]
    return tf.concat([normals[:,:,:,0:2],diffuse, tf.expand_dims(roughness[:,:,:,0], axis=-1), specular], axis=-1)


def deprocess_outputs(outputs):
    partialOutputedNormals = outputs[:,:,:,0:2] * 3.0 #The multiplication here gives space to generate direction with angle > pi/4
    outputedDiffuse = outputs[:,:,:,2:5]
    outputedRoughness = outputs[:,:,:,5]
    outputedSpecular = outputs[:,:,:,6:9]
    normalShape = tf.shape(partialOutputedNormals)
    #newShape = [normalShape[0], normalShape[1], normalShape[2], 1]
    #normalShape[-1] = 1
    tmpNormals = tf.ones_like(outputs[:,:,:,0:1], dtype= tf.float32)

    normNormals = tf_Normalize(tf.concat([partialOutputedNormals, tmpNormals], axis = -1))
    outputedRoughnessExpanded = tf.expand_dims(outputedRoughness, axis = -1)
    return tf.concat([normNormals, outputedDiffuse, outputedRoughnessExpanded, outputedRoughnessExpanded, outputedRoughnessExpanded, outputedSpecular], axis=-1)

def reshape_tensor_display(tensor, splitAmount, logAlbedo = False, axisToSplit = 3, isMaterial = False):
    tensors_list = tf.split(tensor, splitAmount, axis=axisToSplit)#4 * [batch, 256,256,3]
    if tensors_list[0].get_shape()[1] == 1:
        tensors_list = [tf.squeeze (tensor, axis = 1) for tensor in tensors_list]

    if logAlbedo:
        tensors_list[-1] = tf.pow(tensors_list[-1], 0.4545)
        if not isMaterial:
            tensors_list[1] = tf.pow(tensors_list[1], 0.4545)
        else:
            tensors_list[0] = tf.pow(tensors_list[0], 0.4545)
            tensors_list[2] = tf.pow(tensors_list[2], 0.4545)
    tensors = tf.stack(tensors_list, axis = 1) #[batch, 4,256,256,3]
    shape = tf.shape(tensors)
    newShape = tf.concat([[shape[0] * shape[1]], shape[2:]], axis=0)
    tensors_reshaped = tf.reshape(tensors, newShape)

    return tensors_reshaped

def registerTensorboard(paths, images, nbTargets, loss_value, batch_size, targetsRenderings, outputsRenderings):
        inputs = images[0]
        targets = images[1]
        outputs = images[2]

        targetsList = tf.split(targets, batch_size, axis = 0)
        inputsList = tf.split(inputs, batch_size, axis = 0)

        tf.summary.image("targets", targetsList[0], max_outputs=nbTargets)
        tf.summary.image("inputs", inputsList[0], max_outputs=1)
        tf.summary.image("outputs", outputs, max_outputs=nbTargets)
        tf.summary.scalar("loss", loss_value)
        if not targetsRenderings is None:
            #targetsRenderings is [batchSize,nbRenderings, 256, 256, 3]
            tf.summary.image("targets renderings", tf.unstack(tf.log(targetsRenderings[0] + 0.01), axis=0), max_outputs=9)
            tf.summary.image("outputs renderings", tf.unstack(tf.log(outputsRenderings[0] + 0.01), axis=0), max_outputs=9)

def deprocess_image(input, nbSplit, logAlbedo, axisToSplit, isMaterial = False):
    deprocessed = deprocess(input)
    with tf.name_scope("transform_images"):
        reshaped = reshape_tensor_display(deprocessed, nbSplit, logAlbedo, axisToSplit = axisToSplit, isMaterial = isMaterial)
    with tf.name_scope("convert_images"):
        converted = convert(reshaped)
    return converted

def deprocess_input(input):
    with tf.name_scope("convert_images"):
        converted = convert(input)
    return converted

def deprocess_images(inputs, targets, outputs, gammaCorrectedInputs, nbTargets, logAlbedo):

    converted_targets = deprocess_image(targets, nbTargets, logAlbedo, 1)
    converted_targets_gammad = deprocess_image(targets, nbTargets, True, 1)
    converted_outputs = deprocess_image(outputs, nbTargets, logAlbedo, 3)
    converted_outputs_gammad = deprocess_image(outputs, nbTargets, True, 3)

    inputs = deprocess(inputs)
    converted_inputs = deprocess_input(inputs)
    converted_gammaCorrectedInputs = deprocess_input(gammaCorrectedInputs)


    return converted_inputs, converted_targets, converted_outputs, converted_gammaCorrectedInputs, converted_targets_gammad, converted_outputs_gammad


def display_images_fetches(paths, inputs, targets, gammaCorrectedInputs, outputs, nbTargets, logAlbedo):

    converted_inputs, converted_targets, converted_outputs, converted_gammaCorrectedInputs, converted_targets_gammad, converted_outputs_gammad  = deprocess_images(inputs, targets, outputs, gammaCorrectedInputs, nbTargets, logAlbedo)
    with tf.name_scope("encode_images"):
        display_fetches = {
            "paths": paths,
            "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name="input_pngs"),
            "targets": tf.map_fn(tf.image.encode_png, converted_targets, dtype=tf.string, name="target_pngs"),
            "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="output_pngs"),
            "gammaCorrectedInputs": tf.map_fn(tf.image.encode_png, converted_gammaCorrectedInputs, dtype=tf.string, name="gammaInput_pngs"),
            "outputs_gammad": tf.map_fn(tf.image.encode_png, converted_outputs_gammad, dtype=tf.string, name="gammaOutputs_pngs"),
            "targets_gammad": tf.map_fn(tf.image.encode_png, converted_targets_gammad, dtype=tf.string, name="gammaTargets_pngs"),
        }
    images = [converted_inputs, converted_targets, converted_outputs]
    return display_fetches, images

def convert(image, squeeze=False):
    #if a.aspect_ratio != 1.0:
    #    # upscale to correct aspect ratio
    #    size = [CROP_SIZE, int(round(CROP_SIZE * a.aspect_ratio))]
    #    image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BICUBIC)

    if squeeze:
        def tempLog(imageValue):
            imageValue= tf.log(imageValue + 0.01)
            imageValue = imageValue - tf.reduce_min(imageValue)
            imageValue = imageValue / tf.reduce_max(imageValue)
            return imageValue
        image = [tempLog(imageVal) for imageVal in image]
    #imageUint = tf.clip_by_value(image, 0.0, 1.0)
    #imageUint = imageUint * 65535.0
    #imageUint16 = tf.cast(imageUint, tf.uint16)
    #return imageUint16
    return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

def saveInputs(output_dir, fetches, step):
    image_dir = os.path.join(output_dir, "Traininginputs")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filename = "input" + str(step) + ".png"
    out_path = os.path.join(image_dir, filename)
    contents = fetches["gammaCorrectedInputs"][0]

    with open(out_path, "wb") as f:
        f.write(contents)
    return filename

#TODO: Make sure save_images and append indexes still work
def save_images(fetches, output_dir, batch_size, nbTargets, step=None):
    image_dir = os.path.join(output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    filesets = []

    for i, in_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name, "step": step}
        #fetch inputs

        nbCurrentInput = len(fetches["inputs"])//batch_size # This only works if the nb of rendering is constant in the batch.
        for kind in ["inputs","gammaCorrectedInputs"]:
            fileset[kind] = {}
            for idImage in range(nbCurrentInput):
                fileset[kind][idImage] = save_image(idImage, step, name, kind, image_dir, fetches, nbCurrentInput, i)
        #fetch outputs and targets
        for kind in ["outputs", "targets", "targets_gammad", "outputs_gammad"]:
            fileset[kind] = {}
            for idImage in range(nbTargets):
                fileset[kind][idImage] = save_image(idImage, step, name, kind, image_dir, fetches, nbTargets, i)
        filesets.append(fileset)
    return filesets

def save_image(idImage, step, name, kind, image_dir, fetches, nbImagesToRead, materialID):
    filename = name + "-" + kind + "-" + str(idImage) + "-.png"
    if step is not None:
        filename = "%08d-%s" % (step, filename)
    out_path = os.path.join(image_dir, filename)
    contents = fetches[kind][materialID * nbImagesToRead + idImage]

    with open(out_path, "wb") as f:
        f.write(contents)
    return filename

def append_index(filesets, output_dir, nbTargets, mode, step=False):
    index_path = os.path.join(output_dir, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        titles = []
        titles.append("Outputs")
        titles.append("Ground Truth")
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        index.write("<th>Material Name</th>")
        for title in titles:
            index.write("<th>" + title + "</th>")
        index.write("</tr>\n")

    for fileset in filesets:
        index.write("<tr>")
        if step:
            index.write("<td>%d</td>" % fileset["step"])
        index.write("<td>%s</td>" % (fileset["name"]))

        index.write("<td><img src='images/%s'></td>" % fileset["gammaCorrectedInputs"][0])
        index.write("<td><img src='images/%s'></td>" % fileset["inputs"][0])

        index.write("</tr>")

        maps = ["normal", "diffuse", "roughness", "specular"]
        for idImage in range(nbTargets):
            index.write("<tr>")

            index.write("<td>%s</td>" % (maps[idImage]))
            index.write("<td><img src='images/%s'></td>" % fileset["outputs_gammad"][idImage])
            if mode != "eval":
                index.write("<td><img src='images/%s'></td>" % fileset["targets_gammad"][idImage])
            index.write("</tr>\n")

    return index_path

def print_trainable():
    for v in tf.trainable_variables():
        print(str(v.name) + ": " + str(v.get_shape()))

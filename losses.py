import tensorflow as tf
import renderer
import helpers

def l1(x, y):
    return tf.reduce_mean(tf.abs(x-y))

def SSIMLoss(x, y, scale):
    ssim = tf.reduce_mean(tf.image.ssim_multiscale(x, y, scale))
    return  1.0 - ssim

epsilonL1 = 0.01
epsilonRender = 0.01

class Loss:
    lossType = "render"
    batchSize = 8
    lossValue = 0
    materialLossValue = 0
    tile_size = 256
    material_size = 128
    lr = 0.00002
    beta1Adam = 0.5
    nbDiffuseRendering = 3
    nbSpecularRendering = 6
    includeDiffuse = False

    outputs = None
    targets = None
    surfaceArray = None
    outputsRenderings = None
    targetsRenderings = None
    trainOp =  None

    def __init__(self, lossType, outputs, targets, tile_size, batchSize, lr, includeDiffuse, nbSpecularRendering, nbDiffuseRendering) :
        self.lossType = lossType
        self.outputs = outputs
        self.targets = targets
        self.tile_size = tile_size
        self.batchSize = batchSize
        self.lr = lr
        self.includeDiffuse = includeDiffuse
        self.nbSpecularRendering = nbSpecularRendering
        self.nbDiffuseRendering = nbDiffuseRendering

    def __l1Loss(self, outputs, targets):
        #outputs have shape [?, height, width, 12]
        #targets have shape [?, height, width, 12]
        outputsNormal = outputs[:,:,:,0:3]
        outputsDiffuse = tf.log(epsilonL1 + helpers.deprocess(outputs[:,:,:,3:6]))
        outputsRoughness = outputs[:,:,:,6:9]
        outputsSpecular = tf.log(epsilonL1 + helpers.deprocess(outputs[:,:,:,9:12]))

        targetsNormal = targets[:,:,:,0:3]
        targetsDiffuse = tf.log(epsilonL1 + helpers.deprocess(targets[:,:,:,3:6]))
        targetsRoughness = targets[:,:,:,6:9]
        targetsSpecular = tf.log(epsilonL1 + helpers.deprocess(targets[:,:,:,9:12]))

        return l1(outputsNormal, targetsNormal) + l1(outputsDiffuse, targetsDiffuse) + l1(outputsRoughness, targetsRoughness) + l1(outputsSpecular, targetsSpecular)

    def __generateRenderings(self, renderer, batchSize, targets, outputs, surfaceArray):
        diffuses = helpers.tf_generateDiffuseRendering(batchSize, self.nbDiffuseRendering, targets, outputs, renderer)
        speculars = helpers.tf_generateSpecularRendering(batchSize, self.nbSpecularRendering, surfaceArray, targets, outputs, renderer)
        targetsRendered = tf.concat([diffuses[0],speculars[0]], axis = 1)
        outputsRendered = tf.concat([diffuses[1],speculars[1]], axis = 1)
        return targetsRendered, outputsRendered

    def __renderLoss(self, tile_size, batchSize, targets, outputs):
        surfaceArray = helpers.generateSurfaceArray(tile_size)

        rendererImpl = renderer.GGXRenderer(includeDiffuse = self.includeDiffuse)
        targetsRenderings, outputsRenderings = self.__generateRenderings(rendererImpl, batchSize, targets, outputs, surfaceArray)

        reshapedTargetsRendering = tf.reshape(targetsRenderings, [-1, int(targetsRenderings.get_shape()[2]), int(targetsRenderings.get_shape()[3]), int(targetsRenderings.get_shape()[4])])
        reshapedOutputsRendering = tf.reshape(outputsRenderings, [-1, int(outputsRenderings.get_shape()[2]), int(outputsRenderings.get_shape()[3]), int(outputsRenderings.get_shape()[4])])
        currentLoss = l1(tf.log(reshapedTargetsRendering + epsilonRender), tf.log(reshapedOutputsRendering + epsilonRender))

        ssimLoss = SSIMLoss(tf.log(reshapedTargetsRendering + epsilonRender), tf.log(reshapedOutputsRendering + epsilonRender), 1.0)
        lossTotal = currentLoss + ssimLoss
        return lossTotal, targetsRenderings, outputsRenderings

    def __mixedLoss(self, tile_size, batchSize, targets, outputs, lossL1Factor, lossRenderFactor):
        lossVal, targetRenderings, outputsRenderings = self.__renderLoss(tile_size, batchSize, targets, outputs)
        return lossVal + (lossL1Factor * self.__l1Loss(outputs, targets)), targetRenderings, outputsRenderings

    def createLossGraph(self):
        if self.lossType == "render":
            self.lossValue, self.targetRenderings, self.outputsRenderings = self.__renderLoss(self.tile_size, self.batchSize, self.targets, self.outputs)
        elif self.lossType == "l1":
            self.lossValue = self.__l1Loss(self.outputs, self.targets)
        elif self.lossType == "mixed":
            self.lossValue, self.targetRenderings, self.outputsRenderings = self.__mixedLoss(self.tile_size, self.batchSize, self.targets, self.outputs, 0.1, 1.0)

        else:
            raise ValueError('No such loss: ' + self.lossType)

    def createTrainVariablesGraph(self, reuse_bool = False):
        global_step = tf.train.get_or_create_global_step()
        tf.summary.scalar("lr", self.lr)
        with tf.name_scope("model_train"):
            with tf.variable_scope("model_train0", reuse=reuse_bool):
                gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("trainableModel/")]
                gen_optim = tf.train.AdamOptimizer(self.lr, self.beta1Adam)
                gen_grads_and_vars = gen_optim.compute_gradients(self.lossValue, var_list=gen_tvars)
                gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

        ema = tf.train.ExponentialMovingAverage(decay=0.99)
        update_losses = ema.apply([self.lossValue])
        self.lossValue = ema.average(self.lossValue)
        incr_global_step = tf.assign(global_step, global_step+1)
        self.trainOp = tf.group(update_losses, incr_global_step, gen_train)


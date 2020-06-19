from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf
import numpy as np
import argparse
import json
import glob
import random
import collections
import math
import time
import dataReader
import model as mod
import losses
import helpers
import shutil
from random import shuffle

#!!!If running TF v > 2.0 uncomment those lines (also remove the tensorflow import on line 5):!!!
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

#Under MIT License

#Source code tested for tensorflow version 1.12

parser = argparse.ArgumentParser()

parser.add_argument("--mode", required=(__name__ == '__main__'), choices=["train", "test", "finetune"])
parser.add_argument("--output_dir", required=(__name__ == '__main__'), help="where to put output files")
parser.add_argument("--input_dir", help="path to xml file containing information images")
parser.add_argument("--seed", type=int)
parser.add_argument("--checkpoint", default=None, help="directory with checkpoint to resume training from or use for testing")

parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
parser.add_argument("--max_epochs", type=int, help="number of training epochs")
parser.add_argument("--summary_freq", type=int, default=50, help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=1000, help="display progress every progress_freq steps")

parser.add_argument("--save_freq", type=int, default=10000, help="save model every save_freq steps, 0 to disable")
parser.add_argument("--test_freq", type=int, default=20000, help="test model every test_freq steps, 0 to disable")
parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")

parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
parser.add_argument("--which_direction", type=str, default="AtoB", choices=["AtoB", "BtoA"])
parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
parser.add_argument("--input_size", type=int, default=588, help="Size of the input data before cropping")
parser.add_argument("--test_input_size", type=int, default=512, help="Size of the test input data before cropping")

parser.add_argument("--lr", type=float, default=0.00002, help="initial learning rate for adam")
parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
parser.add_argument("--nbTargets", type=int, default=1, help="Number of images to output")
parser.add_argument("--nbInputs", type=int, default=1, help="Number of images in the input")


parser.add_argument("--loss", type=str, default="render", choices=["l1", "render", "mixed"], help="Which loss to use instead of the L1 loss")
parser.add_argument("--nbDiffuseRendering", type=int, default=3, help="Number of diffuse renderings in the rendering loss")
parser.add_argument("--nbSpecularRendering", type=int, default=6, help="Number of specular renderings in the rendering loss")
parser.add_argument("--useLog", dest="useLog", action="store_true", help="Use the log for input")
parser.set_defaults(useLog=False)
parser.add_argument("--includeDiffuse", dest="includeDiffuse", action="store_true", help="Include the diffuse term in the specular renderings of the rendering loss ?")
parser.set_defaults(includeDiffuse=False)

parser.add_argument("--logOutputAlbedos", dest="logOutputAlbedos", action="store_true", help="Log the output albedos")
parser.set_defaults(logOutputAlbedos=False)

parser.add_argument("--imageFormat", type=str, default="png", choices=["jpg", "png", "jpeg", "JPG", "JPEG", "PNG"], help="Which format have the input files")
parser.add_argument("--inputMode", type=str, default="auto", choices=["auto", "xml", "folder", "image", "pythonList"], help="What kind of input to expect")
parser.add_argument("--trainFolder", type=str, default="train", help="train folder to read")
parser.add_argument("--testFolder", type=str, default="test", help="test folder to read")

parser.add_argument("--feedMethod", type=str, default="render", choices=["files", "render"], help="Which feeding method to use")
parser.add_argument("--renderingScene", type=str, default="staticViewPlaneLight", choices=["staticViewPlaneLight", "staticViewSpotLight", "staticViewHemiSpotLight", "staticViewHemiSpotLightOneSurface", "movingViewHemiSpotLightOneSurface", "fixedAngles", "collocatedHemiSpotLightOneSurface", "diffuse", "moreSpecular"], help="Static view with plane light")

parser.add_argument("--jitterLightPos", dest="jitterLightPos", action="store_true", help="Jitter or not the light pos.")
parser.set_defaults(jitterLightPos=False)
parser.add_argument("--jitterViewPos", dest="jitterViewPos", action="store_true", help="Jitter or not the view pos.")
parser.set_defaults(jitterViewPos=False)
parser.add_argument("--useAmbientLight", dest="useAmbientLight", action="store_true", help="use ambient lighting in the rendering, this is not the system used in the fine tuning.")
parser.set_defaults(useAmbientLight=False)
parser.add_argument("--NoAugmentationInRenderings", dest="NoAugmentationInRenderings", action="store_true", help="Use the max pooling system.")
parser.set_defaults(NoAugmentationInRenderings=False)
parser.add_argument("--testApproach", type=str, default="render", choices=["files", "render"], help="Which feeding method to use")



a = parser.parse_args()

if __name__ == '__main__':
    if a.inputMode == "auto":
        if a.input_dir.lower().endswith(".xml"):
            a.inputMode = "xml"
            print("XML Not supported anymore")
        elif os.path.isdir(a.input_dir):
            a.inputMode = "folder"
        else:
            a.inputMode = "image"

TILE_SIZE = 512
inputpythonList = []

def main():
    if a.seed is None:
        a.seed = random.randint(0, 2**31 - 1)

    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    loadCheckpointOption(a.mode, a.checkpoint) #loads so that I don't mix up options and it generates data corresponding to this training

    config = tf.ConfigProto()

    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))

    data = dataReader.dataset(a.input_dir, imageFormat = a.imageFormat, trainFolder = a.trainFolder, testFolder = a.testFolder, nbTargetsToRead = a.nbTargets, tileSize=TILE_SIZE, inputImageSize=a.input_size, batchSize=a.batch_size, fixCrop = (a.mode == "test"), mixMaterials = (a.mode == "train" or a.mode == "finetune"), logInput = a.useLog, useAmbientLight = a.useAmbientLight, useAugmentationInRenderings = not a.NoAugmentationInRenderings)
    # Populate data
    data.loadPathList(a.inputMode, a.mode, a.mode == "train" or a.mode == "finetune", inputpythonList)

    if a.feedMethod == "render":
        if a.mode == "train":
            data.populateInNetworkFeedGraph(a.renderingScene, a.jitterLightPos, a.jitterViewPos,  shuffle = (a.mode == "train"  or a.mode == "finetune"))
        elif a.mode == "finetune":
            data.populateInNetworkFeedGraphSpatialMix(a.renderingScene, shuffle = False, imageSize = a.input_size)

    elif a.feedMethod == "files":
        data.populateFeedGraph(shuffle = (a.mode == "train"  or a.mode == "finetune"))


    if a.mode == "train" or a.mode == "finetune":
        with tf.name_scope("recurrentTest"):
            dataTest = dataReader.dataset(a.input_dir, imageFormat = a.imageFormat, testFolder = a.testFolder, nbTargetsToRead = a.nbTargets, tileSize=TILE_SIZE, inputImageSize=a.test_input_size, batchSize=a.batch_size, fixCrop = True, mixMaterials = False, logInput = a.useLog, useAmbientLight = a.useAmbientLight, useAugmentationInRenderings = not a.NoAugmentationInRenderings)
            dataTest.loadPathList(a.inputMode, "test", False, inputpythonList)
            if a.testApproach == "render":
                #dataTest.populateInNetworkFeedGraphSpatialMix(a.renderingScene, shuffle = False, imageSize = TILE_SIZE, useSpatialMix=False)
                dataTest.populateInNetworkFeedGraph(a.renderingScene, a.jitterLightPos, a.jitterViewPos, shuffle = False)
            elif a.testApproach == "files":
                dataTest.populateFeedGraph(False) 

    targetsReshaped = helpers.target_reshape(data.targetBatch)

    #CreateModel
    model = mod.Model(data.inputBatch, generatorOutputChannels=9)
    model.create_model()
    if a.mode == "train" or a.mode == "finetune":
        testTargetsReshaped = helpers.target_reshape(dataTest.targetBatch)

        testmodel = mod.Model(dataTest.inputBatch, generatorOutputChannels=9, reuse_bool=True)

        testmodel.create_model()
        display_fetches_test, _ = helpers.display_images_fetches(dataTest.pathBatch, dataTest.inputBatch, dataTest.targetBatch, dataTest.gammaCorrectedInputsBatch, testmodel.output, a.nbTargets, a.logOutputAlbedos)

        loss = losses.Loss(a.loss, model.output, targetsReshaped, TILE_SIZE, a.batch_size, tf.placeholder(tf.float64, shape=(), name="lr"), a.includeDiffuse, a.nbSpecularRendering, a.nbDiffuseRendering)

        loss.createLossGraph()
        loss.createTrainVariablesGraph()

    #Register Renderings And Loss In Tensorflow
    display_fetches, converted_images = helpers.display_images_fetches(data.pathBatch, data.inputBatch, data.targetBatch, data.gammaCorrectedInputsBatch, model.output, a.nbTargets, a.logOutputAlbedos)
    if a.mode == "train":
        helpers.registerTensorboard(data.pathBatch, converted_images, a.nbTargets, loss.lossValue, a.batch_size, loss.targetsRenderings, loss.outputsRenderings)

    #Run either training or test
    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])
    saver = tf.train.Saver(max_to_keep=1)
    
    if a.checkpoint is not None:
        print("reading model from checkpoint : " + a.checkpoint)
        checkpoint = tf.train.latest_checkpoint(a.checkpoint)
        partialSaver = helpers.optimistic_saver(checkpoint) #Be careful this will silently not load variables if they are missing from the graph or checkpoint
        
    logdir = a.output_dir if a.summary_freq > 0 else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)

    with sv.managed_session("", config= config) as sess:
        sess.run(data.iterator.initializer)
        print("parameter_count =", sess.run(parameter_count))

        if a.checkpoint is not None:
            print("restoring model from checkpoint : " + a.checkpoint)
            partialSaver.restore(sess, checkpoint)

        max_steps = 2**32
        if a.max_epochs is not None:
            max_steps = data.stepsPerEpoch * a.max_epochs
        if a.max_steps is not None:
            max_steps = a.max_steps

        sess.run(data.iterator.initializer)
        if a.mode == "test":
            filesets = test(sess, data, max_steps, display_fetches, output_dir = a.output_dir)

        if a.mode == "train"  or a.mode == "finetune":
           train(sv, sess, data, max_steps, display_fetches, display_fetches_test, dataTest, saver, loss, a.output_dir)



def loadCheckpointOption(mode, checkpoint):
    if mode == "test":
        if checkpoint is None:
            raise Exception("checkpoint required for test mode")
    
    if not checkpoint is None:
        # load some options from the checkpoint
        options = {"which_direction", "ngf", "nbTargets", "loss", "useLog", "includeDiffuse"}
        with open(os.path.join(checkpoint, "options.json")) as f:
            for key, val in json.loads(f.read()).items():
                if key in options:
                    print("loaded", key, "=", val)
                    setattr(a, key, val)

    for k, v in a._get_kwargs():
        print(k, "=", v)

def test(sess, data, max_steps, display_fetches, output_dir = a.output_dir):
    #testing at most, process the test data once
    sess.run(data.iterator.initializer)
    max_steps = min(data.stepsPerEpoch, max_steps)
    filesets = []
    for step in range(max_steps):
        try:
            results = sess.run(display_fetches)
            filesets.extend(helpers.save_images(results, output_dir, a.batch_size, a.nbTargets))

        except tf.errors.OutOfRangeError:
            print("testing fails in OutOfRangeError")
            continue
    index_path = helpers.append_index(filesets, output_dir, a.nbTargets, a.mode)
    return filesets

def train(sv, sess, data, max_steps, display_fetches, display_fetches_test, dataTest, saver, loss, output_dir = a.output_dir):
    sess.run(data.iterator.initializer)
    try:
        # training
        start_time = time.time()

        for step in range(max_steps):
            options = None
            run_metadata = None
            if helpers.should(a.trace_freq, max_steps, step):
                options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

            fetches = {
                "train": loss.trainOp,
                "global_step": sv.global_step,
            }

            if helpers.should(a.progress_freq, max_steps, step) or step <= 1:
                fetches["loss_value"] = loss.lossValue

            if helpers.should(a.summary_freq, max_steps, step):
                fetches["summary"] = sv.summary_op

            fetches["display"] = display_fetches
            try:
                currentLrValue = a.lr
                if a.checkpoint is None and step < 500:
                    currentLrValue = step * (0.002) * a.lr # ramps up to a.lr in the 2000 first iterations to avoid crazy first gradients to have too much impact.

                results = sess.run(fetches, feed_dict={loss.lr: currentLrValue}, options=options, run_metadata=run_metadata)
            except tf.errors.OutOfRangeError :
                print("training fails in OutOfRangeError, probably a problem with the iterator")
                continue

            global_step = results["global_step"]
            
            #helpers.saveInputs(a.output_dir, results["display"], step)

            if helpers.should(a.summary_freq, max_steps, step):
                sv.summary_writer.add_summary(results["summary"], global_step)

            if helpers.should(a.trace_freq, max_steps, step):
                print("recording trace")
                sv.summary_writer.add_run_metadata(run_metadata, "step_%d" % global_step)

            if helpers.should(a.progress_freq, max_steps, step):
                # global_step will have the correct step count if we resume from a checkpoint
                train_epoch = math.ceil(global_step / data.stepsPerEpoch)
                train_step = global_step - (train_epoch - 1) * data.stepsPerEpoch
                imagesPerSecond = global_step * a.batch_size / (time.time() - start_time)
                remainingMinutes = ((max_steps - global_step) * a.batch_size)/(imagesPerSecond * 60)
                print("progress  epoch %d  step %d  image/sec %0.1f" % (train_epoch, global_step, imagesPerSecond))
                print("Remaining %0.1f minutes" % (remainingMinutes))
                print("loss_value", results["loss_value"])

            if helpers.should(a.save_freq, max_steps, step):
                print("saving model")
                try:
                    saver.save(sess, os.path.join(output_dir, "model"), global_step=sv.global_step)
                except Exception as e:
                    print("Didn't manage to save model (trainining continues): " + str(e))

            if helpers.should(a.test_freq, max_steps, step) or global_step == 1:
                outputTestDir = os.path.join(a.output_dir, str(global_step))
                try:
                    test(sess, dataTest, max_steps, display_fetches_test, outputTestDir)
                except Exception as e:
                    print("Didn't manage to do a recurrent test (trainining continues): " + str(e))

            if sv.should_stop():
                break
    finally:
        saver.save(sess, os.path.join(output_dir, "model"), global_step=sv.global_step) #Does the saver saves everything still ?
        sess.run(data.iterator.initializer)
        outputTestDir = os.path.join(a.output_dir, "final")
        test(sess, dataTest, max_steps, display_fetches_test, outputTestDir )

if __name__ == '__main__':
    main()

def runNetwork(inputDir, outputDir, checkpoint, inputMode = "image", feedMethod = "files", mode="test", input_size=512, nbTargets = 4, batch_size = 1, fileList = [], nbStepMax = 3000, testApproach = "render"):
    inputpythonList.clear()
    a.inputMode = inputMode
    a.feedMethod = feedMethod
    a.input_dir = inputDir
    a.output_dir = outputDir
    a.checkpoint = checkpoint
    a.mode = mode
    a.input_size = input_size
    a.nbTargets = nbTargets
    a.batch_size = batch_size
    a.renderingScene = "diffuse"
    a.max_steps = nbStepMax
    a.save_freq = 100000
    a.test_freq = 1500
    a.progress_freq = 500
    a.loss = "mixed"
    a.lr = 0.00002
    a.useLog = True
    a.summary_freq = 100000
    a.jitterLightPos = True
    a.jitterViewPos = True
    a.includeDiffuse = True
    a.testApproach = testApproach
    a.test_input_size = 512
    
    inputpythonList.extend(fileList)
    tf.reset_default_graph()
    print(a)
    #setup all options...
    main()

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tfHelpers
import helpers
class Model:
    output = None
    inputTensor = None
    ngf = 64 #Number of filter for the generator
    generatorOutputChannels = 9
    reuse_bool=False
    last_convolutions_channels =[64,32,9]

    def __init__(self, input, ngf=64, generatorOutputChannels=9, reuse_bool=False):
        self.inputTensor = input
        self.ngf = ngf
        self.generatorOutputChannels = generatorOutputChannels
        self.reuse_bool = reuse_bool

    def create_model(self):
        with tf.variable_scope("trainableModel", reuse=self.reuse_bool) as scope:
            generator_output, secondary_output = self.create_generator(self.inputTensor, self.generatorOutputChannels, None, self.reuse_bool)
            self.output = helpers.deprocess_outputs(generator_output)

    def GlobalToGenerator(self, inputs, channels):
        with tf.variable_scope("GlobalToGenerator1"):
            fc1 = tfHelpers.fullyConnected(inputs, channels, False, "fc_global_to_unet" ,0.01)
        return tf.expand_dims(tf.expand_dims(fc1, axis = 1), axis=1)

    def logTensor(tensor):
        return  (tf.log(tf.add(tensor,0.01)) - tf.log(0.01)) / (tf.log(1.01)-tf.log(0.01))
        
    def create_generator(self, generator_inputs, generator_outputs_channels, materialEncoded, reuse_bool = True):
        with tf.variable_scope("generator", reuse=reuse_bool) as scope:
            layers = []
            #Input here should be [batch, 256,256,3]
            inputMean, inputVariance = tf.nn.moments(generator_inputs, axes=[1, 2], keep_dims=False)
            globalNetworkInput = inputMean
            globalNetworkOutputs = []

            with tf.variable_scope("globalNetwork_fc_1"):
                globalNetwork_fc_1 = tfHelpers.fullyConnected(globalNetworkInput, self.ngf * 2, True, "globalNetworkLayer" + str(len(globalNetworkOutputs) + 1))
                globalNetworkOutputs.append(tf.nn.selu(globalNetwork_fc_1))

            #encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
            with tf.variable_scope("encoder_1"):
                output = tfHelpers.conv(generator_inputs, self.ngf , stride=2)
                layers.append(output)
            #Default ngf is 64
            layer_specs = [
                self.ngf * 2, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
                self.ngf * 4, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
                self.ngf * 8, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
                self.ngf * 8, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
                self.ngf * 8, # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
                self.ngf * 8, # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
                #self.ngf * 8, # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
            ]

            for layerCount, out_channels in enumerate(layer_specs):
                with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
                    rectified = tfHelpers.lrelu(layers[-1], 0.2)
                    # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
                    convolved = tfHelpers.conv(rectified, out_channels, stride=2)
                    #here mean and variance will be [batch, 1, 1, out_channels]
                    outputs, mean, variance = tfHelpers.instancenorm(convolved)

                    outputs = outputs + self.GlobalToGenerator(globalNetworkOutputs[-1], out_channels)
                    with tf.variable_scope("globalNetwork_fc_%d" % (len(globalNetworkOutputs) + 1)):
                        nextGlobalInput = tf.concat([tf.expand_dims(tf.expand_dims(globalNetworkOutputs[-1], axis = 1), axis=1), mean], axis = -1)
                        globalNetwork_fc = ""
                        if layerCount + 1 < len(layer_specs) - 1:
                            globalNetwork_fc = tfHelpers.fullyConnected(nextGlobalInput, layer_specs[layerCount + 1], True, "globalNetworkLayer" + str(len(globalNetworkOutputs) + 1))
                        else :
                            globalNetwork_fc = tfHelpers.fullyConnected(nextGlobalInput, layer_specs[layerCount], True, "globalNetworkLayer" + str(len(globalNetworkOutputs) + 1))

                        globalNetworkOutputs.append(tf.nn.selu(globalNetwork_fc))
                    layers.append(outputs)

            with tf.variable_scope("encoder_8"):
                rectified = tfHelpers.lrelu(layers[-1], 0.2)
                 # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
                convolved = tfHelpers.conv(rectified, self.ngf * 8, stride=2)
                convolved = convolved  + self.GlobalToGenerator(globalNetworkOutputs[-1], self.ngf * 8)

                with tf.variable_scope("globalNetwork_fc_%d" % (len(globalNetworkOutputs) + 1)):
                    mean, variance = tf.nn.moments(convolved, axes=[1, 2], keep_dims=True)
                    nextGlobalInput = tf.concat([tf.expand_dims(tf.expand_dims(globalNetworkOutputs[-1], axis = 1), axis=1), mean], axis = -1)
                    globalNetwork_fc = tfHelpers.fullyConnected(nextGlobalInput, self.ngf * 8, True, "globalNetworkLayer" + str(len(globalNetworkOutputs) + 1))
                    globalNetworkOutputs.append(tf.nn.selu(globalNetwork_fc))

                layers.append(convolved)
            #default nfg = 64
            layer_specs = [
                (self.ngf * 8 , 0.5),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
                (self.ngf * 8, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
                (self.ngf * 8 , 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
                (self.ngf * 8 , 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
                (self.ngf * 4, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
                (self.ngf * 2 , 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
                (self.ngf , 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
            ]

            num_encoder_layers = len(layers)
            for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
                skip_layer = num_encoder_layers - decoder_layer - 1
                with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
                    if decoder_layer == 0:
                        # first decoder layer doesn't have skip connections
                        # since it is directly connected to the skip_layer
                        input = layers[-1]
                    else:
                        input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

                    rectified = tfHelpers.lrelu(input, 0.2)
                    # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
                    output = tfHelpers.deconv(rectified, out_channels)
                    output, mean, variance = tfHelpers.instancenorm(output)
                    output = output + self.GlobalToGenerator(globalNetworkOutputs[-1], out_channels)
                    with tf.variable_scope("globalNetwork_fc_%d" % (len(globalNetworkOutputs) + 1)):
                        nextGlobalInput = tf.concat([tf.expand_dims(tf.expand_dims(globalNetworkOutputs[-1], axis = 1), axis=1), mean], axis = -1)
                        globalNetwork_fc = tfHelpers.fullyConnected(nextGlobalInput, out_channels, True, "globalNetworkLayer" + str(len(globalNetworkOutputs) + 1))
                        globalNetworkOutputs.append(tf.nn.selu(globalNetwork_fc))
                    if dropout > 0.0:
                        output = tf.nn.dropout(output, rate=dropout)

                    layers.append(output)

            # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
            with tf.variable_scope("decoder_1"):
                input = tf.concat([layers[-1], layers[0]], axis=3)
                rectified = tfHelpers.lrelu(input, 0.2)
                output = tfHelpers.deconv(rectified, generator_outputs_channels)
                lastGlobalNet = self.GlobalToGenerator(globalNetworkOutputs[-1], generator_outputs_channels)
                output = output + lastGlobalNet
                output = tf.tanh(output)
                layers.append(output)

            return layers[-1], lastGlobalNet
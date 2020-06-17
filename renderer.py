import tensorflow as tf
import helpers
import math
import numpy as np

class GGXRenderer:
    includeDiffuse = True

    def __init__(self, includeDiffuse = True):
        self.includeDiffuse = includeDiffuse

    def tf_diffuse(self, diffuse, specular):
        return diffuse * (1.0 - specular) / math.pi

    def tf_D(self, roughness, NdotH):
        alpha = tf.square(roughness)
        underD = 1/tf.maximum(0.001, (tf.square(NdotH) * (tf.square(alpha) - 1.0) + 1.0))
        return (tf.square(alpha * underD)/math.pi)

    def tf_F(self, specular, VdotH):
        sphg = tf.pow(2.0, ((-5.55473 * VdotH) - 6.98316) * VdotH);
        return specular + (1.0 - specular) * sphg

    def tf_G(self, roughness, NdotL, NdotV):
        return self.G1(NdotL, tf.square(roughness)/2) * self.G1(NdotV, tf.square(roughness)/2)

    def G1(self, NdotW, k):
        return 1.0/tf.maximum((NdotW * (1.0 - k) + k), 0.001)


    def tf_calculateBRDF(self, svbrdf, wiNorm, woNorm, currentConeTargetPos, currentLightPos, multiLight, lossRendering):

        h = helpers.tf_Normalize(tf.add(wiNorm, woNorm) / 2.0)
        diffuse = tf.expand_dims(helpers.squeezeValues(helpers.deprocess(svbrdf[:,:,:,3:6]), 0.0,1.0), axis = 1)
        normals = tf.expand_dims(svbrdf[:,:,:,0:3], axis=1)
        normals = helpers.tf_Normalize(normals)
        specular = tf.expand_dims(helpers.squeezeValues(helpers.deprocess(svbrdf[:,:,:,9:12]), 0.0, 1.0), axis = 1)
        roughness = tf.expand_dims(helpers.squeezeValues(helpers.deprocess(svbrdf[:,:,:,6:9]), 0.0, 1.0), axis = 1)
        roughness = tf.maximum(roughness, 0.001)
        
        #This is the simulation of ambient lighting for fine tuning mostly 
        if not lossRendering:
            diffuse = diffuse + (0.15 * specular)

        if multiLight:
            diffuse = tf.expand_dims(diffuse, axis = 1)
            normals = tf.expand_dims(normals, axis = 1)
            specular = tf.expand_dims(specular, axis = 1)
            roughness = tf.expand_dims(roughness, axis = 1)

        NdotH = helpers.tf_DotProduct(normals, h)
        NdotL = helpers.tf_DotProduct(normals, wiNorm)
        NdotV = helpers.tf_DotProduct(normals, woNorm)

        VdotH = helpers.tf_DotProduct(woNorm, h)

        diffuse_rendered = self.tf_diffuse(diffuse, specular)
        D_rendered = self.tf_D(roughness, tf.maximum(0.0, NdotH))
        G_rendered = self.tf_G(roughness, tf.maximum(0.0, NdotL), tf.maximum(0.0, NdotV))
        F_rendered = self.tf_F(specular, tf.maximum(0.0, VdotH))

        specular_rendered = F_rendered * (G_rendered * D_rendered * 0.25)
        result = specular_rendered
        if self.includeDiffuse:
            result = result + diffuse_rendered
        return result, NdotL

    def tf_Render(self, svbrdf, wi, wo, currentConeTargetPos, tensorboard = "", multiLight = False, currentLightPos = None, lossRendering = True, isAmbient = False, useAugmentation = True):
        wiNorm = helpers.tf_Normalize(wi)
        woNorm = helpers.tf_Normalize(wo)

        result, NdotL = self.tf_calculateBRDF(svbrdf, wiNorm, woNorm, currentConeTargetPos, currentLightPos, multiLight, lossRendering)
        resultShape = tf.shape(result)
        lampIntensity = 1.5
        if not currentConeTargetPos is None:
            currentConeTargetDir = currentLightPos - currentConeTargetPos #currentLightPos should never be None when currentConeTargetPos isn't
            coneTargetNorm = helpers.tf_Normalize(currentConeTargetDir)
            distanceToConeCenter = (tf.maximum(0.0, helpers.tf_DotProduct(wiNorm, coneTargetNorm)))
        if not lossRendering:
            if not isAmbient:
                if useAugmentation:
                    stdDevWholeBatch = tf.exp(tf.random_normal((), mean = -2.0, stddev = 0.5))
                    lampIntensity = tf.abs(tf.random_normal((resultShape[0], resultShape[1], 1, 1, 1), mean = 5.0, stddev = 0.15)) # Creates a different lighting condition for each shot of the nbRenderings. Check for over exposure in renderings

                    autoExposure = tf.exp(tf.random_normal((), mean = np.log(1.0), stddev = 0.05))
                    lampIntensity = lampIntensity * autoExposure
                else:
                    lampIntensity = tf.reshape(tf.constant(5.0), [1, 1, 1, 1, 1])
            else:
                if useAugmentation:
                    lampIntensity = tf.exp(tf.random_normal((resultShape[0], 1, 1, 1, 1), mean = tf.log(0.15), stddev = 0.5)) # No need to make it change for each rendering.
                else:
                    lampIntensity = tf.reshape(tf.constant(0.15), [1, 1, 1, 1, 1])

            if multiLight:
                lampIntensity = tf.expand_dims(lampIntensity, axis = 2) #add a constant dim if using multiLight
        lampFactor = lampIntensity * math.pi
        if not isAmbient:
            if not lossRendering:
                lampDistance = tf.sqrt(tf.reduce_sum(tf.square(wi), axis = -1, keep_dims=True))
                lampFactor = lampFactor * helpers.tf_lampAttenuation_pbr(lampDistance)
                
            if not currentConeTargetPos is None:
                if useAugmentation:
                    exponent = tf.exp(tf.random_normal((), mean=np.log(5), stddev=0.15))
                else:
                    exponent = 5.0
                    if lossRendering:
                        exponent = 2.0
                lampFactor = lampFactor * tf.pow(distanceToConeCenter, exponent)
                print("using the distance to cone center")

        result = result * lampFactor

        result = result * tf.maximum(0.0, NdotL)
        if multiLight:
            result = tf.reduce_sum(result, axis = 2) * 1.0#1 / nb of lights
        if lossRendering:
            result = result / tf.expand_dims(tf.maximum(wiNorm[:,:,:,:,2], 0.001), axis=-1) # This division is to compensate for the cosinus distribution of sampling

        return [result]

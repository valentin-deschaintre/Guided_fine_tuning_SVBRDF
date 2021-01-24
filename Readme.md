# Guided Fine-Tuning for Large-Scale Material Transfer

This repository contains the code for our paper "Guided Fine-Tuning for Large-Scale Material Transfer, Valentin Deschaintre, George Drettakis, Adrien Bousseau. Computer Graphics Forum (Eurographics Symposium on Rendering Conference Proceedings),  jul 2020".

The project webpage can be found here: https://team.inria.fr/graphdeco/projects/large-scale-materials/

**The data for the pre-training can be found on the project webpage.**

## Paper abstract
We present a method to transfer the appearance of one or a few exemplar SVBRDFs to a target image representing similar materials. Our solution is extremely simple: we fine-tune a deep appearance-capture network on the provided exemplars, such that it learns to extract similar SVBRDF values from the target image. We introduce two novel material capture and design workflows that demonstrate the strength of this simple approach. Our first workflow allows to produce plausible SVBRDFs of large-scale objects from only a few pictures. Specifically, users only need take a single picture of a large surface and a few close-up flash pictures of some of its details. We use existing methods to extract SVBRDF parameters from the close-ups, and our method to transfer these parameters to the entire surface, enabling the lightweight capture of surfaces several meters wide such as murals, floors and furniture. In our second workflow, we provide a powerful way for users to create large SVBRDFs from internet pictures by transferring the appearance of existing, pre-designed SVBRDFs. By selecting different exemplars, users can control the materials assigned to the target image, greatly enhancing the creative possibilities offered by deep appearance capture.

The video belows shows some of our results.

[![Results video](https://www.youtube.com/embed/x7xB9aGrn9Y/0.jpg)](https://www.youtube.com/embed/x7xB9aGrn9Y)

## /!\Material model
This method is designed to take gamma corrected large scale input pictures (which is internally linearized assuming gamma 2.2) and output gamma corrected albedos maps of the large scale input picture.

The model used is the one described in our single image capture paper: https://github.com/valentin-deschaintre/Single-Image-SVBRDF-Capture-rendering-loss (similar to Adobe Substance), **changing the rendering model implementation to render the results will cause strong appearance difference** as different implementations use the parameters differently (despite sharing their names, for example diffuse and specular will be controled for light conservation or roughness will be squared)! 

## Software requirements
This code relies on Tensorflow 1.X but can be adapted to TF 2.X with the following compatibility code:
    Replace tensorflow import everywhere by:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    
It is based on python 3.X, numpy, imageio and opencv for python.

## Running the finetuning
**First download the pre trained weights here: https://repo-sam.inria.fr/fungraph/large_scale_materials/saved_weights.zip**
Extract the weights in the same folder as the code.
I included in this repository a script which allows to easily run the method on your exemplar. The images are contained in the folder "dataExample" and the pre trained network weights are in "saved_weights".
python testScript_finalTraining.py

## Re-training the network
python largeScale_net.py --mode train --output_dir $OutputDir --input_dir $inputDir --which_direction AtoB --nbTargets 4 --test_freq 20000 --input_size 2048 --loss mixed --batch_size 2 --lr 0.00002 --max_steps 2000000 --useLog --inputMode folder --feedMethod render --jitterLightPos --jitterViewPos --renderingScene diffuse --includeDiffuse --testApproach render --test_input_size 2048

## Bibtex
If you use our code, please cite our paper:

@Article{DDB20,
  author       = "Deschaintre, Valentin and Drettakis, George and Bousseau, Adrien",
  title        = "Guided Fine-Tuning for Large-Scale Material Transfer",
  journal      = "Computer Graphics Forum (Proceedings of the Eurographics Symposium on Rendering)",
  number       = "4",
  volume       = "39",
  year         = "2020",
  keywords     = "material transfer, material capture, appearance capture, SVBRDF, deep learning, fine tuning",
  url          = "http://www-sop.inria.fr/reves/Basilic/2020/DDB20"
}

Codes demonstrating SHAP explanation adapted to semantic segmentation and described in

Please cite the following paper https://hal.archives-ouvertes.fr/hal-03719597, bibtex format:
```
@inproceedings{dardouillet:hal-03719597,
  TITLE = {{Explainability of Image Semantic Segmentation Through SHAP Values}},
  AUTHOR = {Dardouillet, Pierre and Benoit, Alexandre and Amri, Emna and Bolon, Philippe and Dubucq, Dominique and Cr{\'e}doz, Anthony},
  URL = {https://hal.archives-ouvertes.fr/hal-03719597},
  BOOKTITLE = {{ICPR-XAIE}},
  ADDRESS = {Montreal, Canada},
  YEAR = {2022},
  MONTH = Aug,
  KEYWORDS = {Model Explainability ; Image Segmentation ; Shapley Values ; SAR Images},
  PDF = {https://hal.archives-ouvertes.fr/hal-03719597/file/ICPR22_XIAE_LISTIC.pdf},
  HAL_ID = {hal-03719597},
  HAL_VERSION = {v1},
}
```

A model based on [HardNet-msg](https://arxiv.org/abs/2101.07172) pretrained on the [CityScape](https://www.cityscapes-dataset.com/) dataset is proposed ([Weights download link](https://drive.google.com/drive/folders/1TtDWRVxJxc3P4H9Sp46e5m83-tppFBE6?usp=sharing)). This allows running SHAP explanation adapted to semantic segmentation on urban scene segmentation.

# HOW TO :


## Installation
This demo relies on an appropriate installation of Tensorflow >=2.8 with some other libs. Please consider t

Following commands suppose:
 * the framework is path/to/DeepLearningTools/ that should be adjusted with respect to your own installation
 * user prepared a singularity container as proposed in the framework README.md (path/to/DeepLearningTools/README.md)
 * the pretrained model has been downloaded in child folder demo_model. [Download link](https://drive.google.com/drive/folders/1TtDWRVxJxc3P4H9Sp46e5m83-tppFBE6?usp=sharing). Folder organization should then be path/to/DeepLearningTools/examples/xaie/demo_model/exported_models/1

## Run demo
1) First deploy the trained model on a Tensorflow model server.
Using this framework, server start command is :

```
cd path/to/DeepLearningTools
python3 start_model_serving.py -m path/to/DeepLearningTools/examples/xaie/demo_model -psi path/to/DeepLearningTools/install/tf_server.2.8.0.gpu.sif
```

2) Launch a client that will interact with the model, sending perturbed versions of the input and computing explanation maps:
client command is :

```
cd path/to/DeepLearningTools
singularity run --nv /home/alben/workspace/listic-deeptool/install/tf2_addons.2.8.0.opt.sif examples/xaie/SemanticSegmentation_explainSHAP.py --model_dir /home/alben/workspace/listic-deeptool/examples/xaie/demo_model/res_shap_test
```

3) Have a look at the results written in child folder demo_model/

## Customize your trials
 * User can train its own model using this framework and run the script targeting those new models by adjusting the command lines

 * Pixels of interest can be interactively chosen at the beginning of the client process by clicking on the area/point of interest

## Code limitations

 * This demo code needs some more improvements to (re)activate area of interest selection and so on.

 * Explanations based on input perturbation require a lot of processing. Currently, input preparation, model processing and postprocessing are pipelined to improve overall speed but further improvements are welcome !


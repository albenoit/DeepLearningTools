""" Semantic segmentation explanation

Codes demonstrating SHAP explanation adapted to semantic segmentation and described in

Please cite the following paper https://hal.archives-ouvertes.fr/hal-03719597, bibtex format:
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

HOW TO :
1) First deploy the trained model on a tensorflow model server.
Using this framework, server start command is :
python3 start_model_serving.py -m /home/alben/workspace/listic-deeptool/experiments/tests/Cityscapes_hardnetmsegtrials_kafkaFalse_learningRate0.001_inputChannels3_nbClasses34_smoothedParamsTrue_nbEpoch1000_batchSize64_patchSize512_lossfocaltverskyFocal_class_weightingFalse_federatedfedavg_flClients4_outTypelogits_2022-04-15--19:30:23 -psi /home/alben/workspace/listic-deeptool/install/tf_server.2.8.0.gpu.sif

2) Launch a client that will interract with the model, sending perturbed versions of the input and computing explanation maps:
client command is :
singularity run --nv  /home/alben/workspace/listic-deeptool/install/tf2_addons.2.8.0.opt.sif SemanticSegmentation_explainSHAP.py  --model_dir /home/alben/workspace/listic-deeptool/experiments/tests/Cityscapes_hardnetmsegtrials_kafkaFalse_learningRate0.001_inputChannels3_nbClasses34_smoothedParamsTrue_nbEpoch1000_batchSize64_patchSize512_lossfocaltverskyFocal_class_weightingFalse_federatedfedavg_flClients4_outTypelogits_2022-04-15--19:30:23
"""

#profiling tools
import cProfile, io, pstats, os, sys
from turtle import shape
import pickle
#model explanation code
import numpy as np
from scipy import sparse as sparse_mat  
import time

from Pixel_selection import selectpixel, selectROI
from math import ceil, sqrt
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from skimage.segmentation import mark_boundaries
import shap
import csv
import datetime
import tensorflow as tf
import cv2
import os

import configparser
import helpers.model_serving_tools as srv_comm_tools
import argparse

server_crops_per_batch = 16
nb_classes = 34


def explain(server_conn, model_name, image, respath):
    """Use SHAP method to compute explanations of the prediction made by the model
    SHAP: https://arxiv.org/pdf/1705.07874.pdf
    Args:
    model: a keras object containing the model used for predictions
    image_path: path to the  current input image
    """

    # Important parameters used for SHAP algorithm
    zones_factor = 16  # determines super-pixels per band
    background = 0  # Value used for feature masking
    nsamples = 15000  # number of inference done by SHAP function: 50 (testing, bad results) 12000 (prod, good results)
    computeOnRoi = False  # True for ROI, False for pixel list
    empty_image=np.zeros(image.shape, dtype=image.dtype)
    # function used for shap values estimation
    def f_keras(z):
        print("running SHAP function with input vector of size: ", z.shape)
        # Slicing z into batches
        masklist = []
        for i in range(ceil(z.shape[0] / server_crops_per_batch)):
            masklist.append(z[(server_crops_per_batch * i):(server_crops_per_batch * i) + server_crops_per_batch, :])
        # Prepare for model.predict calls
        predic = []
        #with tf.device('/CPU:0'):
        for iteration, batch in enumerate(masklist):
            print("SHAP Values compilation : iteration {i} of {max}...".format(i=iteration, max=len(masklist)))
            # Masking one batch
            time_start=time.time()
            masked_batch = mask_image_sparse(batch)#mask_image(batch, sliced_image, image, background)
            time_batch=time.time()
            probs_shape = list(masked_batch.shape[:-1])+[34]
            #batch_casted = tf.convert_to_tensor(masked_batch, dtype=tf.uint8)
            # Logits from prediction
            #logits = model.predict(batch_casted)
            #print('masked_batch', masked_batch.shape)
            sample={'input':masked_batch}
            #clientIO.getInputData(iteration)
            model_request=srv_comm_tools.generate_single_request(sample, model_name)
            time_request=time.time()
            #receive answer and convert to numpy array
            srv_answer=server_conn.Predict(model_request, 30)
            #print('Server answer type=', type(srv_answer))
            '''
            with open('message.pkl', 'wb') as file:
                pickle.dump(srv_answer, file)
            '''
            output=srv_comm_tools.deserialize_srv_answer_uint8(srv_answer.outputs['probq']).numpy()#output=tf.make_ndarray(srv_answer.outputs["probq"]) #large timeout
            #output=srv_comm_tools.read_srv_answer2(srv_answer, out_type=tf.float32, shape=probs_shape)#output=tf.make_ndarray(srv_answer.outputs["probq"]) #large timeout
            #print('out', output.shape)
            time_answer=time.time()
            #output=tf.make_ndarray(answer.outputs["probq"])#decode_answer(answer)
            #output = np.reshape(tf.constant(answer.outputs["probq"].int_val).numpy(), probs_shape)/255.
            #output = tf.make_ndarray(answer)
            print('Answer shape', output.shape)
            #2 times slower:output = np.reshape(np.asarray(list(answer.outputs['probq'].int_val)), probs_shape).astype(np.float32)/255.
            time_decode=time.time()
            if computeOnRoi:  # Take mean of prediction from the considered ROI
                for i in range(output.shape[0]):
                    #FIXME: divide by 255 TO BE CHECKED, NOT TESTED
                    predic_sample = cv2.mean(output[i, :, :, :], mask_ROI)[0:2]/255.
                    predic.append(predic_sample)
            else:  # Take pred values from only one pixel
                predic_sample = np.zeros((output.shape[0], 0), dtype=np.float32)
                for p in pixels:
                    predic_sample = np.append(predic_sample, output[:, p[1]+col_offset, p[0]+row_offset, :]/255., axis=-1)
                predic.extend(predic_sample)
            time_finalize=time.time()

            print('Iteration timing: batch preparation',
                            round(time_batch-time_start,2),
                 'request:',round(time_request-time_batch, 2),
                 'answer&decode:', round(time_answer-time_request, 2),
                 #'decode:', round(time_decode-time_answer, 2),
                 'finalize:', round(time_finalize-time_answer, 2))
            #print('predic_sample', predic_sample)
            #print('predic_sample.shape', predic_sample.shape)
            #cv2.imshow('predic_sample', predic_sample*255)
            #cv2.waitKey()
            
        return np.array(predic)

    # generate hexagonal grid on input image
    def hexgrid(imagesize, nbhex):
        image = np.zeros((imagesize, imagesize))
        b = 1 / 2
        a = sqrt(3) / 2
        t = int(512 / (nbhex * (1 + b))) + 1
        increment = 1
        for i in range(nbhex * 2 - 1):
            y = (i - 1) * t * a
            if i % 2 == 0:
                x = t * (1 + b)
            else:
                x = 0

            for j in range((nbhex // 2) + 1):
                image = cv2.fillPoly(image, np.int32([[(x, y), (x + t, y), (x + t * (1 + b), y + t * a),
                                                       (x + t, y + t * 2 * a), (x, y + t * 2 * a),
                                                       (x - t * b, y + t * a)]]), increment)
                increment += 1
                x += t * (2 + 2 * b)
        return image

    # create different masks for the input image
    def mask_image(zs, segmentation, image, background):
        out = np.zeros((zs.shape[0], image.shape[0], image.shape[1], image.shape[2]))
        for i in range(zs.shape[0]):
            out[i, :, :, :] = image
            for j in range(image.shape[-1]):
                for k in range(zs.shape[1]):
                    if zs[i, k] == 0:
                        out[i, segmentation[:, :, j] == k] = background
        return out

    def mask_image_sparse(zs):
        print('zs.shape', zs.shape)
        
        def single_masking(single_zs):
            #print('single_zsiiiii', single_zs.astype(bool), single_zs.shape)
            # combine selected masks in a single image mask
            single_zs = np.array(single_zs)
            mask_combi=masks_array[single_zs].sum(axis=0)
            #CHECK tf.sparse.reduce_sum(masks_array[np.where(single_zs!=0)[0]]
            #CHECK tf.sparse.segment_sum(
            if isinstance(mask_combi, int): # manage empty mask set case
                #print('empty...', mask_combi)
                masked_image=empty_image
            else:
                masks=mask_combi.todense().astype(np.uint8)
                #print('masks min, max', masks.min(), masks.max())
                masks=np.expand_dims(masks, -1) #set to 2D+depth shape
                masked_image=image*masks #mask image (broadcast masks on depth dimension)
            return masked_image

        zs=zs.astype(bool).tolist()
        out=np.array(list(map(single_masking, zs))) #force map evaluation with list() seems mandatory...
        '''
        # debug/masks visualization
        print('out', out, out.shape)
        for i in range(out.shape[0]):
            cv2.imshow(str(i), out[i])
        cv2.waitKey()
        '''
        return out

    def save_important_data(pixels, shap_values):
        with open(os.path.join(explanationDir,'shap_values.npy'), 'wb') as file:
            np.save(file, shap_values)
        with open(os.path.join(explanationDir,'shap_phi0.npy'), 'wb') as file:
            np.save(file, explainer.expected_value)
        with open(os.path.join(explanationDir,'explain_map.npy'), 'wb') as file:
            np.save(file, explain_map)
        with open(os.path.join(explanationDir,'colormap.npy'), 'wb') as file:
            np.save(file, cm)
        with open(os.path.join(explanationDir,'hexgrid.npy'), 'wb') as file:
            np.save(file, superpixels)
        with open(os.path.join(explanationDir,'masks_array.npy'), 'wb') as file:
            np.save(file, masks_array)

        # Save important values as a csv file
        filename_csv = 'tab_values.csv'
        filename_sar = "input_boundaries.png"

        shap_values_save = []
        for cl in shap_values:
            shap_values_save.append(cl[0].tolist())
            shap_values_save[-1].insert(0, 'SHAP values')
        result_csv = []
        it = 0
        for px in pixels:
            print('processing pixel', px)
            if computeOnRoi:
                result_csv.append(["VALUES FOR ROI:", ROI_pts])
            else:
                result_csv.append(["VALUES FOR PIXEL IN POSITION (x,y):", px[0], px[1]])
            for cl in range(nb_classes):
                result_csv.append(['class index:', cl])
                result_csv.append(shap_values_save[it])
                result_csv.append(['Expected Value:', explainer.expected_value[it]])
                result_csv.append(['Predicted value:', explainer.expected_value[it] + sum(shap_values[it][0])])
                result_csv.append([])
                it += 1
        f = open(explanationDir + filename_csv, 'w')
        with f:
            writer = csv.writer(f)
            for row in result_csv:
                writer.writerow(row)


        # Saving SAR input + boundaries
        disp_image = mark_boundaries(image / np.max(image), np.asarray(sliced_image[:, :, 0], np.int8), (1., 1., 1.))
        if computeOnRoi:
            mask_ROI_color = cv2.fillPoly(np.zeros(disp_image.shape, np.uint8), [points_poly], (0.0, 0.3, 1.0))
            disp_image = cv2.addWeighted(src1=np.float32(disp_image), alpha=0.8, src2=np.float32(mask_ROI_color), beta=0.2, gamma=0)
        else:
            for px in pixels:
                disp_image = cv2.circle(disp_image, px, 7, (0, 0.3, 1), 2)
        cv2.imwrite(explanationDir + filename_sar, disp_image * 255)

    def save_SHAP_map(numpx, pixel):
        # Filename
        dir_img = "SHAP_map_pixel_{p1}-{p2}/".format(p1=pixel[0], p2=pixel[1])
        output_folder=os.path.join(explanationDir, dir_img)
        os.makedirs(output_folder)
        if computeOnRoi:
            dir_img = "SHAP_map_ROI/"

        print('Saving shap maps...')
        #print('save_SHAP_map: numpx', numpx, 'explain_map.shape', explain_map.shape)
        for it in range(nb_classes):
            #print('it', it)
            plt.imshow(explain_map[numpx * nb_classes + it], cmap=cm)
            plt.savefig(os.path.join(output_folder, f"class_{it}.png"))
        
        # # SHAP subplots
        # fig, axes = plt.subplots(ncols=nb_classes, nrows=2, figsize=(5 * nb_classes, 8))
        #
        # probs_slick_display = cv2.cvtColor(np.float32(outframe_crop[:, :, 1]), cv2.COLOR_GRAY2BGR)
        # probs_seep_display = cv2.cvtColor(np.float32(outframe_crop[:, :, -1]), cv2.COLOR_GRAY2BGR)
        #
        # # Display different information whether Shap is computed on pixel or ROI
        # if computeOnRoi:
        #     predvalue = cv2.mean(outframe_crop[:, :, 1], mask_ROI)[0]
        #     fig.suptitle("Mean of prediction for slick: {prob}".format(prob=format(predvalue, '.3f')))
        #
        #     sar_display = cv2.cvtColor(np.float32(image / np.amax(image)), cv2.COLOR_GRAY2BGR)
        #     mask_ROI_color = cv2.fillPoly(np.zeros(sar_display.shape, np.uint8), [points_poly], (0.0, 0.0, 1.0))
        #     sar_display = cv2.addWeighted(src1=sar_display, alpha=0.8,
        #                                   src2=mask_ROI_color.astype(np.float32), beta=0.2, gamma=0)
        #
        #     probs_slick_display = cv2.addWeighted(src1=probs_slick_display, alpha=0.8,
        #                                           src2=mask_ROI_color.astype(np.float32), beta=0.2, gamma=0)
        #     probs_seep_display = cv2.addWeighted(src1=probs_seep_display, alpha=0.8,
        #                                          src2=mask_ROI_color.astype(np.float32), beta=0.2, gamma=0)
        # else:
        #     predvalue = np.argmax(outframe_crop[pixel[1], pixel[0]])  # for predicted class index
        #     fig.suptitle("Predicted class: {pred} (prob={prob})".format(
        #         pred=predvalue,
        #         prob=format(outframe_crop[pixel[1], pixel[0], predvalue], '.3f')))
        #
        #     sar_display = cv2.cvtColor(np.float32(image / np.amax(image)), cv2.COLOR_GRAY2BGR)
        #     sar_display = cv2.circle(sar_display, pixel, 7, (1, 0.2, 0), 2)
        #
        #     probs_slick_display = cv2.circle(probs_slick_display, pixel, 7, (1, 0.2, 0), 2)
        #     probs_seep_display = cv2.circle(probs_seep_display, pixel, 7, (1, 0.2, 0), 2)
        #
        # # Display shap maps
        # max_val = np.max([np.max(np.abs(i)) for i in shap_values[numpx * nb_classes:(numpx + 1) * nb_classes]])  # max value of SHAP
        # disp = None
        # for it in range(nb_classes):
        #     disp = axes[0, it].imshow(explain_map[numpx * nb_classes + it, :, :, 0], cmap=cm, vmin=-max_val, vmax=max_val)
        #     axes[0, it].imshow(image, cmap='gray', alpha=0.15)
        #     axes[0, it].get_yaxis().set_visible(False)
        #     axes[0, it].get_xaxis().set_visible(False)
        #     axes[0, it].set_title("SHAP({cl})".format(cl=self.classesNames[it]))
        #
        # # display Pred & input image
        # axes[-1, 0].imshow(probs_slick_display)
        # axes[-1, 0].get_yaxis().set_visible(False)
        # axes[-1, 0].get_xaxis().set_visible(False)
        # axes[-1, 0].set_title("Model output for class Slick")
        #
        # axes[-1, 1].imshow(probs_seep_display)
        # axes[-1, 1].get_yaxis().set_visible(False)
        # axes[-1, 1].get_xaxis().set_visible(False)
        # axes[-1, 1].set_title("Model output for class Seep")
        #
        # axes[-1, -1].imshow(sar_display)
        # axes[-1, -1].get_xaxis().set_visible(False)
        # axes[-1, -1].get_yaxis().set_visible(False)
        # axes[-1, -1].set_title("Input image")
        #
        # # colorbar & margin settings for SHAP images
        # cb = fig.colorbar(disp, ax=axes[0, -1], label="importance on prediction", aspect=60, fraction=0.25)
        # cb.outline.set_visible(False)
        # fig.subplots_adjust(top=0.92, bottom=0.005, left=0.005, right=0.995, hspace=0.1, wspace=0.0)
        # fig.savefig(explanationDir + dir_img)

    # Create Folder
    explanationDir = os.path.join(respath, "SHAP_results", "")
    if not os.path.exists(explanationDir):
        os.mkdir(explanationDir)

    # ################# slice the image in areas, for explanation
    print("Hexagonal super-pixel creation")
    superpixels = hexgrid(image.shape[0], zones_factor).astype(np.int16)
    sliced_image = np.tile(superpixels[:, :, None], [1, 1, 3]).astype(np.int16)
    #nbsegments = np.max(sliced_image) + 1 #can get wrong if some superpixels are empty

    def get_masks_array(superpixels):
        masks_list=[]
        for k in range(superpixels.max()+1):
            rows, cols=np.where(superpixels[:, :] == k)
            if len(rows)==0:
                print('empty superpixel:', k)
            else:
                data=[1]*len(rows)
                masks_list.append(sparse_mat.coo_matrix((data, (rows, cols)), shape=superpixels.shape, dtype=bool).tocsr())
        masks_arr=np.array(masks_list)
        return masks_arr
    masks_array=get_masks_array(superpixels)
    print('masks_array', masks_array[0].shape, masks_array[0].dtype)
    nbsegments=masks_array.shape[0]
    print(masks_array[0].todense().shape)
    # copy segmentation on all the channels considered by the model
    #sliced_image = np.ones(image.shape, np.int16) * superpixels
    #nbsegments = np.max(sliced_image) + 1
    print("Number of features (superpixels) covering the image:", nbsegments)

    # ################# tbd
    print("Running SHAP function on the whole image")
    if computeOnRoi:
        ROI_pts = selectROI(image / 255.0)
        print('creating ROI with points: ', ROI_pts)
        points_poly = np.array(ROI_pts, np.int32)
        points_poly = points_poly.reshape((-1, 1, 2))
        mask_ROI = np.uint8(cv2.fillPoly(np.zeros(image.shape[:-1], np.uint8), [points_poly], (1, 1, 1)))
        pixels = [(255, 255)]
    else:
        # aachen_000000_000019_leftImg8bit.png case: pixels = [(245, 282), (371, 266), (359, 369), (428, 219), (233, 152)]
        # aachen_000001_000019_leftImg8bit.png case: pixels = [(390, 170), (237, 253), (158, 373), (176, 201), (107, 188), (363, 93), (209, 117)]
        # frankfurt_000000_000576_leftImg8bit
        pixels = [(253, 186), (142, 260), (130, 188), (355, 158), (351, 331), (353, 232), (360, 272)]
        #pixels = selectpixel(image / 255.0)
    
    argmax_map_RGB = np.tile(argmax_map[:, :, None], [1, 1, 3]).astype(np.uint8)*7
    cv2.imwrite(os.path.join(output_path,'argmax_map.png'), argmax_map*7)
    superpixels_rois=sliced_image.copy().astype(np.uint8)*255
    for px in pixels:
        print('px', px)
        px_output=(px[0]+col_offset, px[1]+row_offset)
        print('px_output', px_output)
        print('px offsets', (col_offset, row_offset))
        argmax_map_RGB = cv2.circle(argmax_map_RGB, px_output, 7, (0, 0, 255), 2)
        superpixels_rois = cv2.circle(superpixels_rois, px, 7, (0, 0, 255), 2)
    cv2.imshow('Class prediction and RoIs', argmax_map_RGB)
    cv2.imshow(str(nbsegments)+' superpixels',superpixels_rois.astype(np.uint8))
    cv2.imwrite(os.path.join(output_path,'argmax_map_circled.png'), argmax_map_RGB)
    cv2.waitKey()
    print('Selected pixels:', pixels)
    print('Press a key to continue...')
    # ################# create explainer of the function f and shap values based on the image
    data = np.zeros((1, nbsegments), dtype=np.int16)
    print('data', data.shape, data.dtype)

    model=f_keras
    #print('model', model.shape, model.dtype)
    explainer = shap.KernelExplainer(model, data)
    print("explainer base values: ", explainer.expected_value)

    shap_values = explainer.shap_values(np.ones((1, nbsegments)), nsamples=nsamples)
    print('Shap values computed, generating explanation maps')
    # ################# buid maps showing shap values for each class
    explain_map = np.zeros((len(shap_values), image.shape[0], image.shape[1]), dtype=np.float32)
    #FIXME, needs to consider masks_array instead of sliced_image (NOT THE SAME NUMBER OF FEATURES) 
    #print('shap_values', shap_values)
    for it, vals in enumerate(shap_values):
        #print('it, vals, vals.shape', it, vals, vals.shape)

        #explain_1class = np.zeros(masks_array[0].shape, np.float32)
        explain_1class = masks_array.copy()
        explain_1class=[ masks_array[feat_id].astype(np.float32)*vals[0,feat_id] for feat_id in range(vals.shape[1])]
        '''for seg in range(nbsegments):
            explain_1class[sliced_image == seg] = vals[0, seg]
        explain_map[it, :, :, :] = explain_1class
        '''
        explain_map[it, :, :] = np.array(explain_1class).sum(axis=0).todense()
    # SHAP map color creation
    colors = []
    for c in np.linspace(1, 0, 100):
        colors.append((245 / 255, 39 / 255, 87 / 255, c))
    for c in np.linspace(0, 1, 100):
        colors.append((24 / 255, 196 / 255, 93 / 255, c))

    cm = LinearSegmentedColormap.from_list("shap", colors)
    print('colormap built, saving maps...')

    print("Saving Results")
    save_important_data(pixels, shap_values)

    try:
        for px_id, px in enumerate(pixels):
            save_SHAP_map(px_id, px)

        print("all files saved in folder ", explanationDir)
    except Exception as e:
        print('Save to figures failed:', e)
        
# retreive command line arguents
parser = argparse.ArgumentParser(description='Deep learning experiments manager')
parser.add_argument("-u","--model_dir",
                    help="path to the target experiment folder")
                    
FLAGS = parser.parse_args()

#get experiment settings filename path
server=srv_comm_tools.get_model_server_cfg(FLAGS.model_dir)['SERVER']
stub = srv_comm_tools.setup_model_server_connexion(server['host'], server['port'],int(1e9) )#experiment_settings.grpc_max_message_length)
"""
#modelpath = '/home/pierre/Documents/TOBETESTED/Cityscapes_hardnetmsegtrials_learningRate0.001_inputChannels3_nbClasses34_smoothedParamsTrue_nbEpoch1000_batchSize64_patchSize512_lossfocaltverskyFocal_class_weightingFalse_federatedfedavg_flClients4_2022-03-08--18:19:07'
modelpath = '/home/alben/workspace/listic-deeptool/experiments/tests/27march/Cityscapes_hardnetmsegtrials_kafkaFalse_learningRate0.001_inputChannels3_nbClasses34_smoothedParamsTrue_nbEpoch1000_batchSize64_patchSize512_lossfocaltverskyFocal_class_weightingFalse_federatedfedavg_flClients4_2022-03-27--13:22:20'
import glob
lastest_save = sorted(glob.glob(modelpath + "/checkpoints/model_epoch*"))[-1]  # folder of the lastest save
print('latest model:', lastest_save)
module_loaded = tf.keras.models.load_model(modelpath + "/checkpoints", compile=True)
print('model OK')
"""
#imgpath = '/home/alben/workspace/listic-deeptool/datasamples/semantic_segmentation/raw_data/aachen_000001_000019_leftImg8bit.png'#aachen_000000_000019_leftImg8bit.png'
imgpath = '/home/alben/workspace/Datasets/CityScapes/leftImg8bit_trainvaltest/leftImg8bit/val/frankfurt/frankfurt_000000_001016_leftImg8bit.png'
raw_img = cv2.imread(imgpath)
if not(isinstance(raw_img, np.ndarray)):
    raise ValueError('Input image could not be loaded')
#rw images case study img = raw_img[200:712, 512:1024, :]
#frankfurt val image case study
img = raw_img[200:712, 512:1024, :]
print('Input image loaded')
cv2.imshow('Original input image', img)

#predict on the original input:
sample={'input':np.expand_dims(img, 0)}
#clientIO.getInputData(iteration)
model_request=srv_comm_tools.generate_single_request(sample, server['model_name'])
#receive and reshape answer
answer=stub.Predict(model_request, 20)
probq=srv_comm_tools.deserialize_srv_answer_uint8(answer.outputs['probq']).numpy()[0]

row_offset=(probq.shape[0]-img.shape[0])//2
col_offset=(probq.shape[1]-img.shape[1])//2
print('img shape', img.shape, probq.shape)
print('probq', probq.shape)
print('prediction map offsets (>0 if predicting only in the central part)', row_offset, col_offset)
#probq = tf.make_ndarray(answer.outputs['probq'])[0]
argmax_map=np.argmax(probq, axis=-1)
cv2.imshow('Class prediction on the entire image', (argmax_map*7).astype(np.uint8))     
output_path=os.path.join(FLAGS.model_dir,'res_shap_test', datetime.datetime.now().strftime("%Y-%m-%d--%H:%M:%S"))
os.makedirs(output_path)

#save base data for offline analysis
with open(os.path.join(output_path,'img.npy'), 'wb') as file:
            np.save(file, img)
with open(os.path.join(output_path,'argmax_map.npy'), 'wb') as file:
            np.save(file, argmax_map)
with open(os.path.join(output_path,'probq_map.npy'), 'wb') as file:
            np.save(file, probq)

pr = cProfile.Profile()
pr.enable()  # start profiling
explain(stub, server['model_name'], img, output_path)
print('Model explanation done')

pr.disable() # profiling end
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
ps.print_stats()
rem = os.path.normpath(os.path.join(os.getcwd(), "..", "..", ".."))
res = s.getvalue().replace(rem, "")
res = res.replace(sys.base_prefix, "").replace("\\", "/")
ps.dump_stats("explain.prof")
print(res)


"""
<pre>Model explanation done
         450558461 function calls (431671328 primitive calls) in 10954.484 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.096    0.096 10954.484 10954.484 SemanticSegmentation_explainSHAP.py:38(explain)
        1    0.051    0.051 10571.346 10571.346 /local/lib/python3.8/dist-packages/shap/explainers/_kernel.py:108(shap_values)
        1    0.164    0.164 10571.283 10571.283 /local/lib/python3.8/dist-packages/shap/explainers/_kernel.py:204(explain)
        3 2877.392  959.131 10460.043 3486.681 SemanticSegmentation_explainSHAP.py:53(f_keras)
        1    0.075    0.075 10458.749 10458.749 /local/lib/python3.8/dist-packages/shap/explainers/_kernel.py:503(run)
      940    0.164    0.000 5568.160    5.924 /local/lib/python3.8/dist-packages/tensorflow/python/framework/tensor_util.py:565(MakeNdarray)
      940 5567.926    5.923 5567.926    5.923 {built-in method numpy.fromiter}
      940    3.335    0.004 1546.195    1.645 /local/lib/python3.8/dist-packages/grpc/_channel.py:937(__call__)
      940    0.007    0.000 1542.857    1.641 /local/lib/python3.8/dist-packages/grpc/_channel.py:919(_blocking)
      940  988.640    1.052  988.652    1.052 {method &apos;next_event&apos; of &apos;grpc._cython.cygrpc.SegregatedCall&apos; objects}
     1880    0.003    0.000  551.511    0.293 /local/lib/python3.8/dist-packages/grpc/_common.py:81(_transform)
      940    0.021    0.000  550.352    0.585 /local/lib/python3.8/dist-packages/grpc/_channel.py:135(_handle_event)
      940    0.002    0.000  550.326    0.585 /local/lib/python3.8/dist-packages/grpc/_common.py:96(deserialize)
      940  550.322    0.585  550.322    0.585 {built-in method FromString}
</pre>
"""
'''
@author : Alexandre Benoit, LISTIC lab, FRANCE (plus some colleagues and interns such as Louis Klein on spring 2017)
@brief  : a set of tools to preprocess data and build up input data pipelines
'''

# python 2&3 compatibility management
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six

#### WARNING, you may have to remove one of the cv2 or gdal import depending on your machine compatibility
import cv2
try:
  from osgeo import gdal
except:
  print('WARNING, could not load GDAL library, this will impact your data pipeline if willing to use it')
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
import time
import copy
import tensorflow as tf
import unicodedata

dataprovider_namescope="data_input_pipeline"
filenames_separator='###'

def make_images_coarse(input_images, downscale_factor=2):
    """
    downscale and upscale a batch of images using nearest neighbors interpolation
    to make those images "coarse"
    @param input_images: the original images to make coarse (expecting 4D tensor)
    @param downscale_factor: the downscaling factor to apply
    @return a batch of images of the same size as the input but made coarser spatially
    """
    #downscale first
    init_height=input_images.get_shape().as_list()[1]
    init_width=input_images.get_shape().as_list()[2]
    new_height=init_height/downscale_factor
    new_width=init_width/downscale_factor
    downscaled=tf.image.resize_nearest_neighbor(
                                        input_images,
                                        size=[new_height, new_width],
                                        align_corners=True,
                                        name='reference_downscale'
                                    )
    #upscals back to initial resolution
    coarse_reference=tf.image.resize_nearest_neighbor(
                                        downscaled,
                                        size=[init_height, init_width],
                                        align_corners=True,
                                        name='reference_coarse'
                                    )

    return coarse_reference

def replace_nans_by_zeros(sample):
    ''' returns a tensor with input nan values replaced by zeros '''
    return tf.where(tf.is_nan(sample), tf.zeros_like(sample), sample)

def plot_sample_channel_histograms(data_sample, filenameID=''):
    ''' Basic data analysis:
    plot the histogram of each channel of a data sample
    @param data_sample, the numpy matrix to process
    @param filenameID, the histogram filename prefix to be used
    '''

    for channelID in six.moves.range(data_sample.shape[-1]):
        plt.figure('Channel ID='+str(channelID))
        plt.hist(data_sample[:,:,channelID])
        plt.title('Channel ID='+str(channelID))
        plt.savefig(str(filenameID)+'RS_dataset_sample_hist_channel'+str(channelID)+'.jpg')

def scaleImg_0_255(img):
    '''simply scale input values to range [0,255] to enable display using OpenCV
    @param, the numpy ndarray to rescale
    @return the rescaled array, type remains the same
    '''
    #get the mask value
    print('Rescaling array of type:'+str(img.dtype))
    #copying before modifying
    img_copy=copy.deepcopy(img)
    try:
        maskValue=np.iinfo(img_copy.dtype).min
        #replace mask values by zeros
        img_copy[img_copy==maskValue]=0
    except:
        print('Failed to detect data type, if float value, then, following should run fine')
    img_min=np.nanmin(img_copy)
    img_max=np.nanmax(img_copy)
    epsilon=1e-4
    scaled_img=((img_copy-img_min)*255.0)/(img_max-img_min+epsilon)
    scaled_img[np.isnan(scaled_img)]=0
    return scaled_img

def debug_show_data(data, message):
    '''used for data debugging:
    @param data, the tensor to show
    @param message, a string to describe the debug message
    use example : tf.py_func(debug_show_data, [raw_sample_crops, 'raw_sample crop'], tf.float32)
    '''
    print("########################################################### DEBUG : {message}:shape={shape}, data={data}".format(message=message, shape=data.shape, data=data))
    return np.float32(1.0)

def extractFilenames(root_dir, file_extension="*.jpg", raiseOnEmpty=True):
    ''' utility function:
    given a root directory and file extension, walk through folderfiles to
    create a list of searched files
    @param root_dir: the root folder from which files should be searched
    @param file_extension: the extension of the files
    @param raiseOnEmpty: a boolean, set True if an exception should be raised if no file is found
    '''
    files  = []
    msg='extractFilenames: from working directory {wd}, looking for files {path} with extension {ext}'.format(wd=os.getcwd(),
                                                                                                                path=root_dir,
                                                                                                                ext=file_extension)
    print(msg)
    for root, dirnames, filenames in os.walk(root_dir):
        file_proto=os.path.join(root, file_extension)
        print('-> Parsing folder : '+file_proto)
        newfiles = glob.glob(file_proto)
        if len(newfiles)>0:
            print('----> Found files:'+str(len(newfiles)))
        files.extend(newfiles)

    if len(files)==0 and raiseOnEmpty is True:
        raise ValueError('No files found at '+msg)
    else:
        print('Found files : '+str(len(files)))
    return sorted(files)

def the_ugly_string_manager(filename):
  ''' horribly ugly code to ensure that a filename string complies with gdal and opencv
      when dealing with python 2 or 3 and pure python or tensorflow py_function
      ... convert to input to unicode to recover properly to string
      FIXME : ... maybe just wait for the death of python2 and hope for a more
      elegant python3
  '''
  if not(isinstance(filename, bytes)):
    filename=bytes(filename, 'utf-8')
  filename_stra=(six.text_type(copy.deepcopy(filename)))
  filename_str=unicodedata.normalize('NFC', filename_stra)
  if filename_str[0]=='b':
    filename_str=filename_str[2:-1]
  return filename_str

def imread_from_gdal(filename, debug_mode=False):
  ''' read an image using OpenCV
      image is loaded as is. In case of a 3 channels image, BGR to RGB conversion
      is applied
      @param filename as a numpy array (coming from Tensorflow)
      @param debug_mode to print more logs on this image read step
  '''
  #get a valid filename string
  filename_str=the_ugly_string_manager(filename)

  ds = gdal.Open(filename_str)
  if ds is None:
      raise ValueError('Could no read file {file}, exists={exists}'.format(file=filename_str,
                                                                           exists=os.path.exists(filename_str)
                                                                           )
                      )
  raster_array=ds.ReadAsArray().transpose([1,2,0])
  if debug_mode == True:
      print('Reading image with GDAL : {file}'.format(file=filename_str))
      print('Image shape='+str(raster_array.shape))
  return raster_array.astype(np.float32)

def imread_from_opencv(filename, cv_imreadMode=-1, debug_mode=False):
  ''' read an image using OpenCV
      image is loaded as is. In case of a 3 channels image, BGR to RGB conversion
      is applied
      @param filename as a numpy array (coming from Tensorflow)
      @param cv_imreadMode as described in the official opencv doc. Note: cv2.IMREAD_UNCHANGED=-1
      @param debug_mode to print more logs on this image read step
  '''

  #get a valid filename string
  filename_str=the_ugly_string_manager(filename)
  image= cv2.imread(filename_str, cv_imreadMode)
  if image is None:
      raise ValueError('Could no read file {file}, exists={exists}'.format(file=filename_str,
                                                                           exists=os.path.exists(filename)
                                                                           )
                      )
  if debug_mode == True:
      print('Reading image with OpenCV: {file} using mode {flag}'.format(file=filename_str, flag=cv_imreadMode))
      print('Image shape='+str(image.shape))
      if len(image.shape)>2:
          print('Image first layer min={minVal}, max={maxVal} (omitting nan values)'.format(minVal=np.nanmin(image[:,:,0]), maxVal=np.nanmax(image[:,:,0])))
      else:
          print('Image first layer min={minVal}, max={maxVal} (omitting nan values)'.format(minVal=np.nanmin(image), maxVal=np.nanmax(image)))
  if len(image.shape)==3: ##reorder channels, from the loaded opencv BGR to tensorflow RGB use
    if image.shape[2]==3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
  return image.astype(np.float32)

def get_sample_entropy(sample):
    ''' @return the entropy of the input tensor
    '''
    with tf.name_scope('sample_entropy'):
        #count unique values occurences
        unique_values, values_idx, counts=tf.unique_with_counts(sample)

        def normalised_entropy(counts):
            classes_prob=tf.math.divide(tf.cast(counts, dtype=tf.float32), tf.cast(tf.reduce_sum(counts), dtype=tf.float32))
            entropy= -tf.reduce_sum(classes_prob*tf.log(classes_prob))
            return tf.math.divide(entropy,tf.log(tf.cast(tf.shape(counts)[0], dtype=tf.float32)))

        #check if more than one class
        normalized_entropy=tf.cond(tf.greater(tf.shape(counts)[0], 1), lambda :normalised_entropy(counts), lambda :0.0)

        #normalized_entropy=tf.Print(normalized_entropy, [counts, normalized_entropy, tf.py_func(get_samples_entropies, [tf.expand_dims(sample,0)], tf.float32)], message='tf_entropyVShandmade')
        return normalized_entropy

def get_sample_entropy_test(values=[0,0,0,1,1,1]):
    ''' test function of get_sample_entropy(sample),
    @param values optionnal parameter to be filled with an array of values
    which entropy is being computed
    '''
    with tf.Session() as sess:
        data=tf.placeholder(dtype=tf.float32, shape=[None])
        entropy_val=sess.run(get_sample_entropy(data), feed_dict={data:values})
        print('Test data='+str(values)+' => Entropy value='+str(entropy_val))

def get_samples_entropies(samples_batch):
    ''' from a batch of 2D image labels, select a subset that ensures a minimum entropy
        @param samples_batch the batch of input data
        @return the vector of size (batchsize) with sample entropy values
    '''
    #print('input crops shape='+str(samples_batch.shape))
    nb_samples=samples_batch.shape[0]
    entropies=np.zeros(nb_samples, dtype=np.float32)
    flatten_samples=np.reshape(samples_batch, [nb_samples, -1]).astype(np.int)

    for it in six.moves.range(nb_samples):
        #print('processing sample '+str(it))
        classes_id_count=np.unique(flatten_samples[it], return_counts=True)
        if len(classes_id_count[0])==1:
            continue
        #print('classes_id_count='+str(classes_id_count))
        classes_prob=classes_id_count[1].astype(float)/float(len(flatten_samples[it]))
        entropies[it]=-(classes_prob*np.log(classes_prob)).sum()/np.log(float(len(classes_prob)))
    return entropies

def convert_semanticMap_contourMap(crops):
    '''
	Convert a semantic map into a contour Map using Sobel operator
    @param crops: the semantic reference map to obtain contour (expecting 4D tensor)

	@return an image batch containing contours of a semantic maps
	'''

    #Border Sobel operator
    sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], tf.float32)
    sobel_x_filter = tf.reshape(sobel_x, [3, 3, 1,1])
    sobel_y_filter = tf.transpose(sobel_x_filter, [1, 0, 2,3])


    image_resized = tf.expand_dims(crops, -1)

    crops_float=tf.cast(image_resized, tf.float32)

    filtered_x = tf.nn.conv2d(crops_float, sobel_x_filter,
                          strides=[1, 1, 1,1], padding='VALID')
    filtered_y = tf.nn.conv2d(crops_float, sobel_y_filter,
                          strides=[1, 1, 1, 1], padding='VALID')

    #sum and threshold
    contours_valid=tf.greater(tf.multiply(filtered_x,filtered_x)+tf.multiply(filtered_y,filtered_y), 1)

    #Add paddings to keep the same shape
    contours = tf.pad(contours_valid, paddings=[[0,0],[1,1],[1,1],[0,0]])

    return contours

class FileListProcessor_Semantic_Segmentation:

    dataprovider_namescope=dataprovider_namescope+'/FileListProcessor_Semantic_Segmentation'

    def __whiten_sample(self, sample):
        ''' apply whitening to a raw+reference image tensorflow
        @param if self.no_reference is False, then input is expected to be a single tensor which first layers are the raw data and the last layer is dense semantic reference
        raw data and reference data are first sliced, whitening is applied on ra data only and finally a single tensor
        is reconstructed and returned. If self.no_reference is True, then the input tensor if considered as a batch on raw data that is being standardized
        @return a tensor similar to the input but which raw data layers have been whitened
        '''
        with tf.name_scope('raw_data_whithening'):
            #apply whitening on the raw data only
            if self.no_reference is False:
                single_image_channels=tf.slice( sample,
                                                begin=[0,0,0],
                                                size=[-1,-1,self.single_image_raw_depth])
                reference_img=tf.slice( sample,
                                                begin=[0,0,self.single_image_raw_depth],
                                                size=[-1,-1,self.single_image_reference_depth])
                reference_img=tf.cast(reference_img, dtype=tf.float32)
                raw_sample=tf.image.per_image_standardization(single_image_channels)
                return tf.concat([tf.cast(single_image_channels, dtype=tf.float32), reference_img], axis=2)
            else:
                return tf.image.per_image_standardization(single_image_channels)

    def __load_raw_ref_images_from_separate_files(self, raw_ref_img_filenames):
        ''' load one raw image and its related reference image and concatenate them into the same image
        images must be of the same size !
        TODO add asserts to heck matching sizes and expected depth
        @param raw_img_filename the filename of the raw image to load
        @param ref_img_filename the filename of the reference image to load
        @return the concatenated image of same 2D size but of depth = raw.depth+ref.depth)
        '''
        with tf.name_scope('read_raw_ref_image_pair'):
            #first split filenames into two strings
            splitted_filenames = tf.strings.split(tf.expand_dims(raw_ref_img_filenames,0), sep=filenames_separator)
            print('splitted_filenames.values[0]=',splitted_filenames.values[0])
            raw_img_filename=splitted_filenames.values[0]
            ref_img_filename=splitted_filenames.values[1]
            if self.use_alternative_imread is not None:
              if self.use_alternative_imread == 'opencv':
                # use Opencv image reading methods WARNING, take care of the channels order that may change !!!
                raw_image = tf.py_func(imread_from_opencv, [raw_img_filename, self.opencv_read_flags], tf.float32, name='raw_data_imread_opencv')
                reference_image = tf.py_func(imread_from_opencv, [ref_img_filename, cv2.IMREAD_GRAYSCALE], tf.float32, name='ref_data_imread_opencv')
                #add a third channel (to be compatible with raw_image rank when willing to concatenate
                reference_image=tf.expand_dims(reference_image, -1)
              elif self.use_alternative_imread == 'gdal':
                # use gdal image reading methods WARNING, take care of the channels order that may change !!!
                raw_image = tf.py_func(imread_from_gdal, [raw_img_filename], tf.float32, name='raw_data_imread_gdal')
                reference_image = tf.py_func(imread_from_gdal, [ref_img_filename], tf.float32, name='ref_data_imread_gdal')
                #add a third channel (to be compatible with raw_image rank when willing to concatenate
                reference_image=tf.expand_dims(reference_image, -1)
            else:
                # use Tensorflow image reading methods
                #-> read raw data
                single_raw_element = tf.read_file(raw_img_filename)
                single_reference_element = tf.read_file(ref_img_filename)
                #decode raw data using a specific decoder
                raw_image=tf.image.decode_png(single_raw_element, channels=self.single_image_raw_depth, dtype=None, name="single_image_raw_read")
                reference_image=tf.image.decode_png(single_reference_element, channels=self.single_image_reference_depth, dtype=None, name="single_image_reference_read")
            print('raw data channels='+str(self.single_image_raw_depth))
            print('dense semantic labels channels='+str(self.single_image_reference_depth))
            #concatenate both images in a single one
            return tf.cast(tf.concat([raw_image, reference_image], axis=2, name='concat_inputs'), dtype=tf.float32, name='to_float')


    def __load_raw_ref_images_from_single_file(self, raw_img_filename):
        ''' load one raw image with its related reference image encoded as the last channel
        @param raw_img_filename the filename of the raw image to load with last channel being the reference semantic data
        @return the concatenated image of same 2D siae but of depth = raw.depth+ref.depth)
        '''
        with tf.name_scope('read_raw_ref_single_image'):
            print('raw_img_filename='+str(raw_img_filename))
            if self.use_alternative_imread == 'opencv':
              # use Opencv image reading methodcropss WARNING, take care of the channels order that may change !!!
              raw_ref_image = tf.py_func(imread_from_opencv, [raw_img_filename, self.opencv_read_flags], tf.float32, name='raw_ref_data_imread_opencv')
            elif self.use_alternative_imread == 'gdal':
              # use GDAL image reading methodcropss WARNING, take care of the channels order that may change !!!
              raw_ref_image = tf.py_func(imread_from_gdal, [raw_img_filename], tf.float32, name='raw_ref_data_imread_gdal')
            else:
              raise ValueError('Neither OpenCV nor GDAL selected to read data and ground truth from the same image')
            print('raw data channels='+str(self.single_image_raw_depth))
            print('dense semantic labels channels='+str(self.single_image_reference_depth))
            #concatenate both images in a single one
            return tf.cast(raw_ref_image, dtype=tf.float32, name='to_float')

    def __generate_crops(self, input_image):
        ''' considering an input tensor of any shape, divide it into overlapping windows and put them into a queue
          inspired from http://stackoverflow.com/questions/40186583/tensorflow-slicing-a-tensor-into-overlapping-blocks
          @param input_image image to be sampled
        '''
        with tf.name_scope('generate_crops'):
            if self.field_of_view > 0:
                radius_of_view = (self.field_of_view-1)//2
            else:
                radius_of_view=0
            with tf.name_scope('prepare_crops_bbox'):
                height=tf.cast(tf.shape(input_image)[0], dtype=tf.int32, name='image_height')
                width=tf.cast(tf.shape(input_image)[1], dtype=tf.int32, name='image_width')

                if self.shuffle_samples == True:
                    self.nbPatches=tf.cast(self.image_area_coverage_factor*tf.cast(height*width, dtype=tf.float32)/tf.constant(self.patchSize*self.patchSize, dtype=tf.float32), dtype=tf.int32, name='number_of_patches')
                    self.nbPatches=tf.minimum(self.nbPatches, tf.constant(self.max_patches_per_image, name='max_patches_per_image'), name='saturate_number_of_patches')
                    random_vector_shape = [self.nbPatches]
                    top_coord = tf.random_uniform(random_vector_shape,0,width-self.patchSize+2*radius_of_view,dtype=tf.int32,name='patch_top_coord_top')
                    left_coord = tf.random_uniform(random_vector_shape,0, height-self.patchSize+2*radius_of_view, dtype=tf.int32,name='patch_left_coord')

                    #manage global image borders padding
                    paddings = [[radius_of_view,self.patchSize],[radius_of_view,self.patchSize],[0,0]]

                    '''debug_1=tf.py_func(debug_show_data, [left_coord, 'flat_meshgrid_x'], tf.float32)
                    debug_2=tf.py_func(debug_show_data, [top_coord, 'flat_meshgrid_y'], tf.float32)
                    '''
                    #with tf.control_dependencies([debug_1, debug_2]):
                    input_image = tf.pad(input_image, paddings)
                else: #expecting TEST dataset use case : no padding, only processing original pixels, avoiding border effects
                    top_coord = tf.range(0, width-self.patchSize,self.patchSize-2*radius_of_view,dtype=tf.int32)
                    left_coord = tf.range(0, height-self.patchSize, self.patchSize-2*radius_of_view,dtype=tf.int32)

                    flat_meshgrid_y, flat_meshgrid_x = tf.meshgrid(top_coord, left_coord)
                    left_coord = tf.reshape(flat_meshgrid_x, [-1])
                    top_coord =  tf.reshape(flat_meshgrid_y, [-1])
                    self.nbPatches = tf.shape(left_coord)[0]

                #normalize coordinates
                left_coord = tf.cast(left_coord, dtype=tf.float32, name='patch_left_coord')
                top_coord = tf.cast(top_coord, dtype=tf.float32, name='patch_top_coord')

                height_norm=tf.cast(height, tf.float32) - 1.
                width_norm=tf.cast(width, tf.float32) - 1.
                print((top_coord, height_norm,width_norm))
                top_coord_normalized=tf.math.divide(top_coord,height_norm)
                left_coord_normalized=tf.math.divide(left_coord,width_norm)
                bottom_coord_normalized=(top_coord+self.patchSize)/height_norm
                right_coord_normalized=(left_coord+self.patchSize)/width_norm
                boxes=tf.stack([top_coord_normalized, left_coord_normalized, bottom_coord_normalized, right_coord_normalized], axis=1)
                print('boxes : '+str(boxes))
                crops=tf.image.crop_and_resize(tf.expand_dims(input_image,0),#get batch to a "batch 4D tensor"
                                               boxes=boxes,
                                               box_ind=tf.zeros(self.nbPatches, dtype=tf.int32),
                                               crop_size=[self.patchSize,self.patchSize],
                                               method='nearest')#FIXME, MUST BE nearest neighbors here !!!'
                print('crops : '+str(crops))

                return tf.data.Dataset.from_tensor_slices(crops)

    def crop_filter(self, crop):
      ''' a tf.data.Dataset filter function
          Args:
           crops: a set of crop candidates
          Returns:
           selected_crops: a vector of size equal to the number of input crops with True for accepted candidates, False if not
      '''
      print('crop_filter input: '+str(crop))
      with tf.name_scope('filter_crops'):
        # default selection value : every crop is selected
        selected_crop = tf.constant(True)
        print('selected_crop INIT='+str(selected_crop))

        #next processing wrt config
        if self.balance_classes_distribution is True  and self.no_reference is False: #TODO second test is a safety test that could be removed is safety test done before
            ref_slice=tf.slice(crop,
                                begin=[0,0,self.single_image_raw_depth],
                                size=[-1,-1,self.single_image_reference_depth])
            selected_crop = tf.logical_and(selected_crop, tf.greater(get_sample_entropy(tf.reshape(ref_slice,[-1])), self.classes_entropy_threshold, name='minimum_labels_entropy_selection'))

        if self.manage_nan_values is 'avoid':
            print('FileListProcessor_Semantic_Segmentation: crops with Nan values will be avoided')
            selected_crop = tf.logical_and(selected_crop, tf.logical_not(tf.reduce_any(tf.is_nan(crop))))#tf.math.logical_and(selected_crops, tf.reduce_any(tf.is_nan(crop_candidate, axis=0)))
        print('selected_crop AFTER='+str(selected_crop))
        #TODO add more selection strategies
        return selected_crop


    def __image_transform(self, input_image):
        ''' apply a set of transformation to an input image
        @param input_image, the image to be transformed. It must be a stack of
        the raw image (first layers) followed by the reference layer(s)
        @return the transformed raw+reference concatenated image, only geometric transforms are applied to the reference image
        '''
        with tf.name_scope('image_transform'):
            print('__image_transform input: '+str(input_image))

            #retreive a single crop
            """ standard cropping scheme """
            transformed_image=input_image
            #apply basic transforms
            if self.manage_nan_values is 'zeros':
                transformed_image = replace_nans_by_zeros(transformed_image)
            if self.apply_random_flip_left_right:
                transformed_image=tf.image.random_flip_left_right(transformed_image)
            if self.apply_random_flip_up_down:
                transformed_image=tf.image.random_flip_up_down(transformed_image)

            single_image_channels=transformed_image #by default apply the folowing transform to all the image channels
            if self.no_reference is False: #if using a reference channel, then apply transform only of the raw data

                #reconstruct the initial input image to adjust its brightness and contrast homogeneously accross channels
                single_image_channels=tf.slice( transformed_image,
                                                begin=[0,0,0],
                                                size=[-1,-1,self.single_image_raw_depth])
                reference_img=tf.slice( transformed_image,
                                                begin=[0,0,self.single_image_raw_depth],
                                                size=[-1,-1,self.single_image_reference_depth])
                reference_img=tf.cast(reference_img, dtype=tf.float32)

            print('single_image_channels crop shape='+str(single_image_channels.get_shape().as_list()))
            if self.apply_random_brightness != None:
                single_image_channels=tf.image.random_brightness(single_image_channels, max_delta=self.apply_random_brightness)
            if self.apply_random_saturation != None:
                single_image_channels=tf.image.random_contrast(single_image_channels, lower=1.0-self.apply_random_saturation, upper=1.0+self.apply_random_saturation)
            if self.apply_whitening:     # Subtract off the mean and divide by the variance of the pixels.
                single_image_channels=tf.image.per_image_standardization(single_image_channels)

            if self.no_reference is False:#get back to the input+reference images concat
                out_image= tf.concat([tf.cast(single_image_channels, dtype=tf.float32), reference_img], axis=2)
            else:
                out_image= tf.cast(single_image_channels, dtype=tf.float32)
            return tf.data.Dataset.from_tensors(out_image)

    def __create_dataset_raw_images_reader(self):
        ''' given the chosen mode (using a list of filename pairs of raw+ref image or using a single filename poiting a single raw+ref(lastchannel) image)
            -> create the dataset object instance that loads the filename lists
        '''
        raw_sample=None
        datasetfiles=None
        if self.image_pairs_raw_ref_input:
            datasetFiles=[''+string1+filenames_separator+string2  for string1,string2 in zip(self.filelist_raw_data, self.filelist_reference_data)]
        else: #raw and ref data in the same image of only raw data use cases
            datasetFiles=self.filelist_raw_data
        #print(datasetFiles)

        #apply general setup for dataset reader : read all the input list one time, shuffle if required to, read one by one
        self.dataset=tf.data.Dataset.from_tensor_slices(datasetFiles)
        if self.shuffle_samples:
              self.dataset=self.dataset.shuffle(len(datasetFiles))


        def __load_raw_images_from_filenames(filenames):
          ''' function to be applied for each of the dataset sample
          '''
          print('single sample input filename(s)='+str(filenames))
          if self.image_pairs_raw_ref_input:
              raw_sample=self.__load_raw_ref_images_from_separate_files(raw_ref_img_filenames=filenames)
          else: #raw and ref data in the same image of only raw data use cases
              raw_sample=self.__load_raw_ref_images_from_single_file(raw_img_filename=filenames)
          if self.full_frame_mode == True:
              self.cropSize=self.fullframe_ref_shape
          else:
              self.cropSize=[self.patchSize,self.patchSize,self.single_image_raw_depth+self.single_image_reference_depth]
          print('Deep net will be fed by samples of shape='+str(self.cropSize))
          return tf.data.Dataset.from_tensors(raw_sample)
        if self.debug:
          print('input filename(s) dataset='+str(self.dataset))
        self.dataset=self.dataset.flat_map(__load_raw_images_from_filenames)


    def getIteratorInitializer(self):
      ''' specify here all the ops to run at the init step '''
      return self.dataset_iterator.initializer

    def __create_data_pipeline(self):
        """ input pipeline is defined on the CPU parameters
        """
        with tf.device("/cpu:0"),tf.name_scope(FileListProcessor_Semantic_Segmentation.dataprovider_namescope+'_gen_crops_then_transform'):
            """create a first "queue" (actually a list) of pairs of image filenames
            and generate data samples (whole read images)
            """
            #1. let the dataset load the raw images and associates possible metadata as last channel(s)
            self.__create_dataset_raw_images_reader()


            #2. transform the dataset samples convert raw images into crops
            if self.full_frame_mode == True:
              with tf.name_scope('full_raw_frame_prefectching'):
                if self.apply_whitening:     # Subtract off the mean and divide by the variance of the pixels.
                    self.dataset=self.dataset.flat_map(self.__whiten_sample)
            else:
                self.dataset=self.dataset.flat_map(self.__generate_crops)

            #dataset prefetch size:
            prefect_size = self.batch_size*self.num_preprocess_threads#, (self.deep_data_queue_capacity*3)/4)

            #finalise the dataset pipeline : filterout
            self.dataset=self.dataset.filter(self.crop_filter).flat_map(self.__image_transform)
            if self.shuffle_samples:
              self.dataset=self.dataset.shuffle(int(self.batch_size*self.max_patches_per_image)) #shuffle prefetch size set empirically high
            #finalize dataset (set nb epoch and batch size)
            self.dataset=self.dataset.repeat(self.nbEpoch).batch(self.batch_size)
            self.dataset_iterator = self.dataset.make_initializable_iterator()

        print('Input data pipeline graph is now defined')

    """
        @brief a class enabling couples of raw Data+ full frame reference samples enqueing
        *special case: if patch_ratio_vs_input==1 and max_patches_per_image==1
        -> then, enqueue the full image with full reference
        *when generating patches, the number of patches is computed like this:
        nbPatches_per_image=min(max_patches_per_image, image_area_coverage_factor*(input image pixels/(patchSize*patchSize))
         last member shows we initially get the optimal number of patches so that the number of pixels of eall the patches is similar to the input image number of pixels
         and this number of patches is multiplied by a factor given by the user. Upper limit is given by max_patches_per_image.
        ==> adjust max_patches_per_image and image_area_coverage_factor to your needs.
         Be aware that max_patches_per_image modulates the size of the samples queue so that large values gives large queues that ensures a good mixing of samples from various images
         Be aware that once all samples are generated for a given image, if balance_classes_distribution is True, then many samples are filtered out to ensure a good entropy level on the labels reference samples

        Parameters:
        filelist_raw_data: the list of raw files to process
        filelist_reference_data: reference data source follwing rule:
                -if provided data is a python list : this will be considered as the list of ground truth associated to the filelist_raw_data (SAME ORDER EXPECTED!)
                -if None, then the reference image is expected to be the last channel of the raw data
                -if -1, then, no reference is expected so that the data provider will only sample raw data and won't provide any reference data (as for unsupervised learning)
        nbEpoch: an integer that specifies the number of expected epoch (-1 or None forces to repeat indefinitely)
        shuffle_samples: set True if samples should be shuffled at the entry of the deep net (typical training use case),
                         set False to preserve patchs ordering on a regular grid NOTE : no zero padding is done, image right/bottom borders may not be sampled
        patch_ratio_vs_input: the size ratio of the crops generated from each input image
                    --> if >=1 || <0, then, max_patches_per_image will be forced to 1
        max_patches_per_image: the number of crops extracted per image
        image_area_coverage_factor: the number of patches per images is automatically computed to get nearly the same number
         of pixels as the input image (surface coverage), this factor is applied
         to this number of patches. However the maximum limit of patches is forced by max_patches_per_image
        num_preprocess_threads: the number of threads run in parallel to generate input data
        apply_random_flip_left_right: set True if input should be randomly mirrored left-right,
        apply_random_flip_up_down: set True if input should be randomly mirrored up-down
        apply_random_brightness: set None is not used, set >0 if brighness should be randomly adjusted by this factor
        apply_random_saturation: set None is not used, set >0 if saturation should be randomly adjusted by this factor
        apply_whitening: set True to whiten RAW DATA ONLY !!!
        batch_size: set the number of sample provided at each consuming step
        use_alternative_imread: set False if data should be loaded from tensorflow image read methods (for now, jpeg and png only)
        balance_classes_distribution: set False if no sample pop out should be applied
                set True if some sample crops should be removed in order to get equally distributed classes
        classes_entropy_threshold: if balance_classes_distribution is True, then use this parameter in range [0,1]
        in order to select crops with higher normalized entropy than this value
        opencv_read_flags: if usig OpenCV to read images, set here the specific cv2.ilread flags for specific image formats
        field_of_view: size of the field of view of a pixel. Used to define the size of the overlap of adjacent crops of an image when not using random crops
        manage_nan_values: set 'zeros' to replace nan values by zeros, 'avoid' to avoid sample crops with nan values, None else and Exception will be raised to highlight potential dataset problems
    """
    def __init__(self, filelist_raw_data,
                    filelist_reference_data,
                    nbEpoch=-1,
                    shuffle_samples=True,
                    patch_ratio_vs_input=0.2,
                    max_patches_per_image=10,
                    image_area_coverage_factor=2.0,
                    num_preprocess_threads=4,
                    apply_random_flip_left_right=True,
                    apply_random_flip_up_down=False,
                    apply_random_brightness=0.5,
                    apply_random_saturation=0.5,
                    apply_whitening=True,
                    batch_size=50,
                    use_alternative_imread=False,
                    balance_classes_distribution=False,
                    classes_entropy_threshold=0.6,
                    opencv_read_flags=-1,#cv2.IMREAD_UNCHANGED=-1, #cv2.IMREAD_LOAD_GDAL | cv2.IMREAD_ANYDEPTH ):
                    field_of_view=0,
                    manage_nan_values=None,
                    debug=False):
        self.filelist_raw_data=filelist_raw_data
        self.filelist_reference_data=filelist_reference_data
        self.nbEpoch=nbEpoch
        self.shuffle_samples=shuffle_samples
        self.patch_ratio_vs_input=patch_ratio_vs_input
        self.max_patches_per_image=max_patches_per_image
        self.image_area_coverage_factor=float(image_area_coverage_factor)
        self.num_preprocess_threads=num_preprocess_threads
        self.apply_random_flip_left_right=apply_random_flip_left_right
        self.apply_random_flip_up_down=apply_random_flip_up_down
        self.apply_random_brightness=apply_random_brightness
        self.apply_random_saturation=apply_random_saturation
        self.apply_whitening=apply_whitening
        self.batch_size=batch_size
        self.use_alternative_imread=use_alternative_imread
        self.balance_classes_distribution=balance_classes_distribution
        self.classes_entropy_threshold=classes_entropy_threshold
        self.opencv_read_flags=opencv_read_flags
        self.field_of_view = field_of_view
        self.manage_nan_values=manage_nan_values
        self.debug = debug

        if self.image_area_coverage_factor<=0:
            raise ValueError('Error when constructing DataProvider: image_area_coverage_factor must be above 0')

        #first read the first raw and reference images to get aspect ratio and depth
        #FIXME : fast change to introduce gdal image loading, TO BE CLARIFIED ASAP !!!
        if self.use_alternative_imread == 'gdal':
          raw0=imread_from_gdal(filelist_raw_data[0], True)
        else:
          raw0=imread_from_opencv(filelist_raw_data[0],opencv_read_flags, True)
        print('Read first raw image {filepath} of shape {shape}'.format(filepath=filelist_raw_data[0], shape=raw0.shape))
        self.single_image_raw_width = raw0.shape[0]
        self.single_image_raw_height = raw0.shape[1]

        #adjusting setup and pipeline architecture design depending on the inputs
        self.image_pairs_raw_ref_input=False
        self.no_reference=False
        if filelist_reference_data == None:
            #-> case of raw images that include reference as the last channel
            print('*** Dataprovider is sampling raw data and expects it to have reference (ground truth) at the last channel')
            self.single_image_raw_depth=raw0.shape[2]-1
            self.single_image_reference_depth=1
            self.fullframe_ref_shape=list(raw0.shape)
        elif type(filelist_reference_data) is list:
            print('*** Dataprovider is sampling raw data and reference data lists')
            self.image_pairs_raw_ref_input=True
            #-> case of raw images plus separate reference images
            ref0=cv2.imread(filelist_reference_data[0],self.opencv_read_flags)
            print('read first reference image {filepath} of shape {shape}'.format(filepath=filelist_reference_data[0], shape=ref0.shape))
            if raw0.shape[:2] != ref0.shape[:2]:
                raise ValueError('FileListProcessor_input::__init__ Error, first input files do not have the same pixel size')
            self.single_image_raw_depth=raw0.shape[2]
            self.single_image_reference_depth=1
            self.fullframe_ref_shape=list(raw0.shape)
            self.fullframe_ref_shape[-1]=self.single_image_raw_depth+self.single_image_reference_depth

            if len(ref0.shape)>2:
                self.single_image_reference_depth=ref0.shape[2]
        else:
            print('*** Dataprovider is sampling raw data but not providing any reference data')
            self.no_reference=True
            self.single_image_raw_depth=raw0.shape[2]
            self.single_image_reference_depth=0
        self.img_ratio=float(raw0.shape[0])/float(raw0.shape[1])

        #parameters check:
        if patch_ratio_vs_input==1 or patch_ratio_vs_input<0:
            print('Each image will be entirely processed')
            self.patch_ratio_vs_input=1
            self.max_patches_per_image=1

        if patch_ratio_vs_input >1:
            self.patchSize=patch_ratio_vs_input
            self.patch_ratio_vs_input=float(patch_ratio_vs_input)/float(raw0.shape[0])
        else:
            self.patchSize=int(raw0.shape[0]*patch_ratio_vs_input)

        self.full_frame_mode=self.patch_ratio_vs_input==self.max_patches_per_image==1
        if self.full_frame_mode == True:
            print('==> each image will finally be processed globally following')
        else:
            print("Image patches will be of size:"+str(self.patchSize))
            print("Image patches ratio vs input is:"+str(self.patch_ratio_vs_input))

        # checking if field_of_view is odd
        if self.field_of_view%2 == 0 and self.field_of_view != 0:
            raise ValueError('field_of_view must be odd or 0 (current : {})'.format(self.field_of_view))

        #create the input pipeline
        self.__create_data_pipeline()


def extract_feature_columns(data_dict, features_labels):
    ''' function that creates feature columns interpreters and applies them on
    the input data to prepare a dense tensor that will feed a model
    @param data_dict: a batch of samples where columns are aggregated into a dictionary
    @param features_labels: a dictionary following architecture:
    features_labels={'all_cols':{'names':colnames, 'record_defaults':record_defaults},
                     'data_cols':{'names_opt_categories_or_buckets':data_cols, 'indexes':data_idx},
                     'labels_cols':{'names':label_cols,'indexes':label_idx}}
    where names_opt_categories_or_buckets is a dictionnary that contains at
    least key 'name' and has an optionnal LIST specified by key :
    ---'vocabulary_list' if column is categorial and should be one hot encoded according to the specified vocabulary
    ---'buckets_boundaries' if column numeric but should be should be one hot encoded according to the specified boundarie values
    ---'normalizer_fn' if the column is numeric and should be normalized
    @returns a dense data tensor
    '''
    #preparing input data features, convert to the appropriate type
    data_features=[]
    for data_col in features_labels['data_cols']['names_opt_categories_or_buckets']:
       print('***preparing input data column:'+str(data_col))
       if len(data_col)==1 or 'normalizer_fn' in data_col:
           print('----->numeric data to be casted as float 32')
           normalization_fn=None
           if 'normalizer_fn' in data_col:
               normalization_fn=data_col['normalizer_fn']
               print('Normalization function found:'+str(data_col['normalizer_fn']))
           data_features.append(tf.feature_column.numeric_column(key=data_col['name'], normalizer_fn=normalization_fn))
       elif 'vocabulary_list' in data_col:
           print('----->categorial data to be one hot encoded')
           #name=data_col['name'].split(' ')[0]+'_indicator'
           data_features.append(tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list(
                        key=data_col['name'],
                        vocabulary_list=data_col['vocabulary_list'],
                        num_oov_buckets=1#allow one other category for value out of the vocabulary
                        )))
       elif 'buckets_boundaries' in data_col:
            print('----->numeric data to be one hot encoded with respect to some boundaries')
            # First, convert the raw input to a numeric column.
            numeric_feature_column = tf.feature_column.numeric_column(key=data_col['name'])

            # Then, bucketize the numeric column on the years 1960, 1980, and 2000.
            data_features.append(tf.feature_column.bucketized_column(
                source_column = numeric_feature_column,
                boundaries = data_col['buckets_boundaries']))
       else:
            raise ValueError('Could not manage the proposed data col')

    print('*** data columns, len='+str(len(data_dict))+' : '+str(data_dict))
    print('*** feature columns, len='+str(len(data_features))+' : '+str(data_features))
    data_vectors = tf.feature_column.input_layer(features=data_dict, feature_columns=data_features)
    print('*** data_vectors='+str(data_vectors))

    return data_vectors

def breakup(x, lookback_len):
  ''' break a sequence into overlapping sub sequences
  Arg:
    x: the input sequence of size N steps of shape[batch_size]
  Returns:
    a stack of overlapping sub sequences
  '''
  #get the sequence length
  N = tf.shape(x)[0]
  windows = [tf.slice(x, [b], [lookback_len]) for b in six.moves.range(0, N-lookback_len)]
  return tf.stack(windows)

def FileListProcessor_csv_time_series(files, csv_field_delim, record_defaults_values, nblines_per_block, queue_capacity, shuffle_batches, na_value_string='N/A', labels_cols_nb=1, device="/cpu:0", breakup_fact=1):
    ''' define a queue that prefetches 1D vectors coming from a set of csv files
        records that can have empty cells but default values must be specified to replace empty spaces
        @param files: the list of input csv files to read from
        @param csv_field_delim: the cell delimiter in the csv file
        @param record_defaults_values: the list of default values, for example: [['default_txt'],[1],[1.0]]
        @param nblines_per_block:the number of lines to read at a time : this will enable to dequeue this number lines as a single datablock
        @param queue_capacity: the number of sample slots available in the queue
        @param na_value_string: the string used in the csv files to say 'abnormal'/'undefined' value
        @param labels_cols_nb: the number of FIRST columns that should be used to generate labels (say if start_date and stop_date exist, this would let this parameter to 2)
        @param device: the device where to place the data pipeline
    '''
    with tf.device(device),tf.name_scope('csv_file_line_blocks_read_enqueue'): # force input pipeline to be ran on the main cpu
        filename_queue = tf.train.string_input_producer(files, num_epochs=None)
        reader = tf.TextLineReader(skip_header_lines=1)
        key, value = reader.read_up_to(filename_queue, num_records=nblines_per_block*breakup_fact, name='read_lines')
        # Default values, in case of empty columns. Also specifies the type of the
        # decoded result.
        features = tf.decode_csv(
            value, record_defaults=record_defaults_values, field_delim=csv_field_delim, na_value=na_value_string) #keep all the the first column (timestamp)
        #stack all but first column in a single tensor
        print('FileListProcessor_csv_time_series: features='+str(features))

        timestamplabels=[tf.cast(features[i], dtype=tf.string) for i in six.moves.range(labels_cols_nb)]
        timestamps=tf.string_join(timestamplabels)
        raw_data_sample=tf.cast(tf.stack(features[labels_cols_nb:], axis=1), dtype=tf.float32)

        print("FileListProcessor_csv_time_series: raw_data_sample="+str(raw_data_sample))
        if breakup_fact>1:
          raw_data_sample=breakup(raw_data_sample, lookback_len=nblines_per_block)
          timestamps=breakup(timestamps, lookback_len=nblines_per_block)
          print("BREAKEDUP raw_data_sample="+str(raw_data_sample))

        '''raw_data_sample = tf.Print(raw_data_sample, [raw_data_sample],
               'read lines = ', summarize=2, first_n=10)'''
        print('FileListProcessor_csv_time_series: timestamps/labels='+str(timestamps))
        #print('features='+str(features))
        #print('raw_data_sample='+str(raw_data_sample))
        #setup a data queue that will prefetch the data
        if shuffle_batches:
            data_queue=tf.RandomShuffleQueue(capacity=queue_capacity,
                                             min_after_dequeue=queue_capacity/2,
                                             dtypes=['string','float'],
                                             shapes=[nblines_per_block,[nblines_per_block, raw_data_sample.get_shape().as_list()[-1]]],
                                             name='shuffled_prefetch_data_queue')
        else:
            data_queue=tf.FIFOQueue(capacity=queue_capacity,
                                   dtypes=['string','float'],
                                   shapes=[nblines_per_block,[nblines_per_block, raw_data_sample.get_shape().as_list()[-1]]],
                                   name='fifo_prefetched_data_queue')

        #enqueue data ONLY if the requested number of lines has been grabbed
        check_nblines_per_block_OK=tf.equal(tf.shape(key)[0],nblines_per_block)
        if breakup_fact==1:
          conditionned_enqueue_op=tf.cond(check_nblines_per_block_OK, lambda: data_queue.enqueue([timestamps,raw_data_sample]), lambda:tf.no_op())
        else:
          conditionned_enqueue_op=tf.cond(check_nblines_per_block_OK, lambda: data_queue.enqueue_many([timestamps,raw_data_sample]), lambda:tf.no_op())

        #threading
        thread_enqueue_data = tf.train.QueueRunner( data_queue,
                                                    [conditionned_enqueue_op]*5)
        tf.train.add_queue_runner(thread_enqueue_data)
        #monitor queues:
        #tf.summary.scalar('raw_crops_queue_size', self.raw_crops_queue.size())
        tf.summary.scalar('data_queue_size', data_queue.size())

        #reset op
        reset_ops=[reader.reset()]
    return data_queue, reset_ops

def FileListProcessor_csv_lines(files, csv_field_delim, shuffle_batches, batch_size, features_labels, na_value_string='N/A', samples_window_size=1, samples_window_shift=1, nbEpoch=1, device="/cpu:0", debug=False):
    ''' inspired by the official iris tutorial and related blog
        https://developers.googleblog.com/2017/12/
        define a queue that prefetches 1D vectors coming from a set of csv files
        records that can have empty cells but default values must be specified to replace empty spaces
        @param files: the list of input csv files to read from
        @param csv_field_delim: the cell delimiter in the csv file
        @param queue_capacity: the number of sample slots available in the queue
        @param shuffle_batches: True to randomize extracted samples
        @param batch_size: the number of elements to extract at each step
        @param features_labels: a dictionary following architecture:
        features_labels={'all_cols':{'names':colnames, 'record_defaults':record_defaults},
                         'data_cols':{'names_opt_categories_or_buckets':data_cols, 'indexes':data_idx},
                         'labels_cols':{'names':label_cols,'indexes':label_idx}}
        where names_opt_categories_or_buckets is a dictionnary that contains at
        least key 'name' and has an optionnal LIST specified by key :
        ---'vocabulary_list' if column is categorial and should be one hot encoded according to the specified vocabulary
        ---'buckets_boundaries' if column numeric but should be should be one hot encoded according to the specified boundarie values


        @param na_value_string: the string used in the csv files to say 'abnormal'/'undefined' value
        @param device: the device where to place the data pipeline
    '''

    with tf.device(device),tf.name_scope('csv_file_lines_read_enqueue'): # force input pipeline to be ran on the main cpu
        def decode_csv(line):
            if debug is True:
                line=tf.Print(line,[line], message='RAW csv line=')
            print('record default=',features_labels['all_cols']['record_defaults'])
            parsed_line = tf.decode_csv(line, record_defaults=features_labels['all_cols']['record_defaults'], field_delim=csv_field_delim, na_value=na_value_string)
            features = dict(zip(features_labels['all_cols']['names'], parsed_line))
            unused_cols = set(features_labels['all_cols']['names']) - {col['name'] for col in features_labels['data_cols']['names_opt_categories_or_buckets']} \
                                                                    - {col for col in features_labels['labels_cols']['names']}
            print('### UNUSED COLUMNS='+str(unused_cols))
            # Remove unused columns
            for col in unused_cols:
                features.pop(col)
            print('### USED COLUMNS (data and labels)='+str(features))
            print('features='+str(features))
            return features

        dataset = (tf.data.TextLineDataset(files)  # Read text file
           .skip(1)  # Skip header row
           .repeat(nbEpoch)
           )
        #decode csv line(s)
        dataset=dataset.map(decode_csv)  # Decode each line in a multi thread mode (asynchronous reading)
        #dataset=tf.data.Dataset.from_tensor_slices(dataset)
        print("====================")  # ==> "(tf.float32, tf.int32)"
        print(dataset.output_types)  # ==> "(tf.float32, tf.int32)"
        print('####################')  # ==> "(tf.float32, tf.int32)"
        print(dataset.output_shapes)
        if samples_window_size>1: # group sets of consecutive lines into batches
           #print('************************************ dataset='+str(dataset))

           dataset=dataset.window(size=samples_window_size, shift=samples_window_shift, stride=1)
           #print('************************************ dataset='+str(dataset))
        print("+++++++++++++++++++++")  # ==> "(tf.float32, tf.int32)"
        print(dataset.output_types)  # ==> "(tf.float32, tf.int32)"
        print(dataset.output_shapes)

        dataset=dataset.prefetch(10*batch_size)  # Make sure you always have 1 batch ready to serve

        if shuffle_batches is True:
            dataset=dataset.shuffle(buffer_size=5*batch_size)

        #create batches of data (blocs)
        dataset=dataset.batch(batch_size)

        return dataset

def FileListProcessor_image_classification(sourceFolder, file_extension,
                                           use_alternative_imread=False,
                                           image_reader_flags=-1,
                                           shuffle_batches=True,
                                           batch_size=1,
                                           device="/cpu:0",
                                           debug=False):
  '''
    Loads a set of images from a folder with associated labels for image classification
    Args:
      sourceFolder : the parent folder of the target files
      file_extension : the target file extension
      use_alternative_imread : set False to read from Tensorflow native ops or set 'gdal' or 'opencv' to read with those tools,
      image_reader_flags : set -1 to use defaults or add specific flags to provide to the image readers
      shuffle_batches : set True to shuffle false to keep order
      batch_size : an indicatr batch size to dimension the prefectch queue
      device : the device where to put the dataset provider (better to set on CPU ("/cpu:0"))
      debug: Boolean, if True, prints additionnal logs
  '''
  ds = tf.data.Dataset.list_files(os.path.join(sourceFolder,file_extension))
  ds = ds.map(map_func=load_image)

  dataset = (tf.data.TextLineDataset(files)  # Read text file
     .skip(1)  # Skip header row
     .map(decode_csv, num_parallel_calls=4)  # Decode each line in a multi thread mode (asynchronous reading)
     .cache() # Warning: Caches entire dataset, can cause out of memory
     .repeat(1)    # Repeats dataset only one time (one epoch) thus allowing to automatically switch between train and eval steps
     .batch(batch_size)
     .prefetch(5*batch_size)  # Make sure you always have 1 batch ready to serve
     )
  if shuffle_batches is True:
      dataset=dataset.shuffle(buffer_size=5*batch_size)

  return dataset

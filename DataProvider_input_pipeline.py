'''
@author : Alexandre Benoit, LISTIC lab, FRANCE (plus some colleagues and interns such as Louis Klein on spring 2017)
@brief  : a set of tools to preprocess data and build up input data pipelines
'''
import cv2
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
import time
import copy
import tensorflow as tf

dataprovider_namescope="data_input_pipeline"

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
    ''' returns a tensor withinput nan values replaced by zeros '''
    return tf.where(tf.is_nan(sample), tf.zeros_like(sample), sample)

def plot_sample_channel_histograms(data_sample, filenameID=''):
    ''' Basic data analysis:
    plot the histogram of each channel of a data sample
    @param data_sample, the numpy matrix to process
    @param filenameID, the histogram filename prefix to be used
    '''

    for channelID in xrange(data_sample.shape[-1]):
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
    scaled_img=((img_copy-img_min)*255.0)/(img_max-img_min)
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

def extractFilenames(root_dir, file_extension="/*.jpg"):
    ''' utility function:
    given a root directory and file extension, walk through folderfiles to
    create a list of searched files
    @param root_dir: the root folder from which files should be searched
    @param file_extension: the extension of the files
    '''
    files  = []
    for root, dirnames, filenames in os.walk(root_dir):
        file_proto=(root+ file_extension)
        print('looking for files '+file_proto)
        files.extend(glob.glob(file_proto))
    return sorted(files)

def imread_from_opencv(filename, cv_imreadMode=cv2.IMREAD_UNCHANGED, debug_mode=False):
  ''' read an image using OpenCV
      image is loaded as is. In case of a 3 channels image, BGR to RGB conversion
      is applied
      @param filename as a numpy array (coming from Tensorflow)
      @param cv_imreadMode as described in the official opencv doc
  '''
  image= cv2.imread(str(filename), cv_imreadMode)
  if image is None:
      raise ValueError('Failed to read image '+filename)
  if debug_mode == True:
      print('Reading image {file} using mode {flag}'.format(file=filename, flag=cv_imreadMode))
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
        #abort if empty
        classes_prob=tf.divide(tf.cast(counts, dtype=tf.float32), tf.cast(tf.reduce_sum(counts), dtype=tf.float32))
        normalized_entropy= tf.divide(-tf.reduce_sum(classes_prob*tf.log(classes_prob)), tf.log(tf.cast(tf.shape(counts)[0], dtype=tf.float32)))
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

    for it in xrange(nb_samples):
        #print('processing sample '+str(it))
        classes_id_count=np.unique(flatten_samples[it], return_counts=True)
        if len(classes_id_count[0])==1:
            continue
        #print('classes_id_count='+str(classes_id_count))
        classes_prob=classes_id_count[1].astype(float)/float(len(flatten_samples[it]))
        entropies[it]=-(classes_prob*np.log(classes_prob)).sum()/np.log(float(len(classes_prob)))
    return entropies

def space_to_batch_overlap(input_image, patch_size, field_of_view):
    ''' from a single image, return a batch of adjacent crops with an overlap
        @param input_image the image to crop in batch
        @param patch_size the size of the crops
        @param field_of_view the diameter of the overlap
        @return a batch of adjacent crop of size (patch_size) with and overlap of (field_of_view-1)/2
    '''
    with tf.name_scope('space_to_batch_overlap'):
        #ease the calculus
        if field_of_view >0:
            radius_of_view = (field_of_view-1)/2
        else:
            radius_of_view=0
        height=tf.cast(tf.shape(input_image)[0], dtype=tf.float32, name='image_height')
        width=tf.cast(tf.shape(input_image)[1], dtype=tf.float32, name='image_width')
        nb_patch_w = tf.cast(tf.ceil((width+2*radius_of_view)/(patch_size-2*radius_of_view)), dtype=tf.int32,name='nb_patch_width')
        nb_patch_h = tf.cast(tf.ceil((height+2*radius_of_view)/(patch_size-2*radius_of_view)), dtype=tf.int32,name='nb_patch_width')
        nb_patch = nb_patch_h*nb_patch_w

        #input_image = tf.image.pad_to_bounding_box(input_image, radius_of_view, radius_of_view, tf.cast(new_heigth, tf.int32), tf.cast(new_width,tf.int32))
        input_image = tf.pad(input_image, [[radius_of_view,radius_of_view+patch_size],[radius_of_view,radius_of_view+patch_size],[0,0]])
        #init the result
        raw_sample_tensor_4d=tf.zeros([0, patch_size,patch_size,tf.shape(input_image)[2]],dtype=tf.uint8)
        #calculate the top/left position of each patch
        top_coord = tf.range(0,width+-patch_size+1+2*radius_of_view,patch_size-2*radius_of_view)
        left_coord = tf.range(0, height+2*radius_of_view, patch_size-2*radius_of_view)
        patches_left, patches_top = tf.meshgrid(top_coord, left_coord)
        patches_top = tf.cast(tf.reshape(patches_top, [-1]),dtype=tf.int32,name='coord_top_of_patch')
        patches_left = tf.cast(tf.reshape(patches_left, [-1]),dtype=tf.int32,name='coord_left_of_patch')
        # prepare the var
        patch_size = tf.cast(patch_size,dtype=tf.int32)
        # index for the loop
        x=tf.constant(0,dtype=tf.int32,name='index')
        # crop each patch with loop
        def body(index, input_image, batch, top_coord, left_coord, patch_size, nb_patch):
            crop = tf.slice(input_image, size=[patch_size, patch_size,-1], begin=[top_coord[index], left_coord[index],0])
            batch = tf.concat([batch, tf.expand_dims(crop,0)],0)
            return index+1, input_image,batch,top_coord,left_coord,patch_size,nb_patch
        #stop after nb_patch crop
        def cond(index, input_image, batch, top_coord, left_coord, patch_size, nb_patch):
            return index<nb_patch
        #_temp are mandatory here, but not used
        x, _temp1,crops, _temp2,_temp3,_temp4, _temp5 = tf.while_loop(cond=cond,body=body,loop_vars=[x,input_image, raw_sample_tensor_4d, patches_top, patches_left, patch_size, nb_patch], shape_invariants=[tf.TensorShape(None), input_image.shape, tf.TensorShape([None, None, None, None]), tf.TensorShape(None), tf.TensorShape(None), tf.TensorShape(None),tf.TensorShape(None)], parallel_iterations=10, name="generating_crops")

    return tf.cast(crops,dtype=tf.float32)

def space_to_batch_queue(input_image, patch_size, fov, queue, standardize_data=False, max_crops_number_exception=1000):
	''' considering an input tensor of any shape, divide it into overlapping windows and put them into a queue
	inspired from http://stackoverflow.com/questions/40186583/tensorflow-slicing-a-tensor-into-overlapping-blocks
	@param input_image image to be sampled
	@param patch_size size of the window
	@param fov size of the overlap
	@param queue the queue the window will be put into.
	'''
	with tf.name_scope('generate_crops'):
		if fov > 0:
			radius_of_view = (fov-1)/2
		else:
			radius_of_view=0

		height=tf.cast(tf.shape(input_image)[0], dtype=tf.float32, name='image_height')
		width=tf.cast(tf.shape(input_image)[1], dtype=tf.float32, name='image_width')
		top_coord = tf.range(0, width+2*radius_of_view,patch_size-2*radius_of_view)
		left_coord = tf.range(0, height+2*radius_of_view, patch_size-2*radius_of_view)

		flat_meshgrid_y, flat_meshgrid_x = tf.meshgrid(top_coord, left_coord)
		flat_meshgrid_x = tf.cast(tf.reshape(flat_meshgrid_x, [-1]),dtype=tf.int32,name='coord_top_of_patch')
		flat_meshgrid_y = tf.cast(tf.reshape(flat_meshgrid_y, [-1]),dtype=tf.int32,name='coord_left_of_patch')
        number_of_crops=flat_meshgrid_x.shape[0]
        #assert_op=tf.Assert(tf.less_equal(number_of_crops, max_crops_number_exception), [number_of_crops], name='ensure_crops_queue_capacity')
        #with tf.control_dependencies([assert_op]):
        crops_topleft_queue = tf.train.slice_input_producer([flat_meshgrid_y, flat_meshgrid_x],
            	                                           shuffle=False,
                                                           capacity=max_crops_number_exception,
                                                           num_epochs=None)
        top_crop = crops_topleft_queue[1]
        left_crop = crops_topleft_queue[0]
        paddings = [[radius_of_view,patch_size+radius_of_view],[radius_of_view,patch_size+radius_of_view],[0,0]]
        input_image = tf.pad(input_image, paddings)

        single_crop = tf.slice(input_image, begin=[top_crop, left_crop, 0], size=[patch_size, patch_size, -1])
        if standardize_data is True:
    	       return queue.enqueue(tf.image.per_image_standardization(single_crop))
        else:
    	       return queue.enqueue(single_crop)

def batch_to_space(crops, patch_size, fov, number_of_patches_height, number_of_patches_width, original_height, original_width):
	'''
	recombine a batch of images into the original image
	@param crops the batch (it needs to be of the form [batch, height, width, depth])
	@param patchSize size of the images in the batch
	@param fov size of the overlap between the patches
	@param number_of_patches_height number of patches in the height dimension
	@param number_of_patches_width number of patches in the width dimension
	@param original_height height of the original that will be returned
	@param original_width width of the original that will be returned
	@return an image constructed from crops
	'''
	# Need to adapt to the way the image is constructed
	with tf.name_scope('batch_to_space'):
		if fov >0:
			radius_of_view = (fov-1)/2
		else:
			radius_of_view=0
		# center the crops (removing fov)
		new_patch_size = patch_size - 2*radius_of_view
		centered_crops=tf.slice(crops, begin=[0,radius_of_view,radius_of_view,0], size=[-1, new_patch_size, new_patch_size, -1])
		# reconstruct the image
		reconstruction = tf.transpose(centered_crops, [0,2,1,3])
		reconstruction = tf.reshape(reconstruction, [-1,new_patch_size,tf.shape(crops)[3]])
		reconstruction = tf.transpose(reconstruction, [1,0,2])
		reconstruction = tf.concat(tf.split(reconstruction, number_of_patches_height, axis = 1 ), axis=0)
		reconstruction = tf.slice(reconstruction,begin=[0,0,0],size=[original_height, original_width,-1])
		return reconstruction

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
    def __load_raw_ref_images(self, raw_img_filename, ref_img_filename):
        ''' load one raw image and its related reference image and concatenate them into the same image
        images must be of the same size !
        TODO add asserts to heck matching sizes and expected depth
        @param raw_img_filename the filename of the raw image to load
        @param ref_img_filename the filename of the reference image to load
        @return the concatenated image of same 2D size but of depth = raw.depth+ref.depth)
        '''
        with tf.name_scope('read_raw_ref_image_pair'):
            if self.use_opencv_imread == True:
                # use Opencv image reading methods WARNING, take care of the channels order that may change !!!
                raw_image = tf.py_func(imread_from_opencv, [raw_img_filename, self.opencv_read_flags], tf.float32, name='raw_data_imread_opencv')
                reference_image = tf.py_func(imread_from_opencv, [ref_img_filename, cv2.IMREAD_GRAYSCALE], tf.float32, name='ref_data_imread_opencv')
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


    def __load_raw_ref_single_image(self, raw_img_filename):
        ''' load one raw image with its related reference image encoded as the last channel
        @param raw_img_filename the filename of the raw image to load with last channel being the reference semantic data
        @return the concatenated image of same 2D siae but of depth = raw.depth+ref.depth)
        '''
        with tf.name_scope('read_raw_ref_single_image'):
            # use Opencv image reading methodcropss WARNING, take care of the channels order that may change !!!
            raw_ref_image = tf.py_func(imread_from_opencv, [raw_img_filename, self.opencv_read_flags], tf.float32, name='raw_ref_data_imread_opencv')
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
                radius_of_view = (self.field_of_view-1)/2
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
                    left_coord = tf.cast(tf.reshape(flat_meshgrid_x, [-1]),dtype=tf.int32,name='patch_top_coord_top')
                    top_coord = tf.cast(tf.reshape(flat_meshgrid_y, [-1]),dtype=tf.int32,name='patch_left_coord')
                    self.nbPatches = tf.shape(left_coord)[0]

            idx=tf.constant(0,dtype=tf.int32,name='index')
            raw_sample_tensor_4d=tf.zeros([0, self.patchSize,self.patchSize,tf.shape(input_image)[2]],dtype=tf.float32)
            # crop each patch with loop
            def body(index, input_image, batch, top_coord, left_coord, patch_size, nb_patch):
                crop =  tf.slice(input_image,
                                size=[patch_size, patch_size,-1],
                                begin=[top_coord[index], left_coord[index],0],
                                name='get_single_crop')
                #manage nan values if required
                if self.manage_nan_values is 'zeros':
                    print('FileListProcessor_Semantic_Segmentation: Nan values replaced by zeros')
                    if self.balance_classes_distribution is True  and self.no_reference is False:
                        ref_slice=tf.slice(crop,
                                            begin=[0,0,self.single_image_raw_depth],
                                            size=[-1,-1,self.single_image_reference_depth])
                        batch = tf.cond(tf.greater(get_sample_entropy(tf.reshape(ref_slice,[-1])), self.classes_entropy_threshold, name='minimum_labels_entropy_selection'),
                                                   lambda : tf.concat([batch, replace_nans_by_zeros(tf.expand_dims(self.__image_transform(crop),0))],0),#if entropy is enough
                                                   lambda : batch)
                    else:
                        batch =  tf.concat([batch, replace_nans_by_zeros(tf.expand_dims(self.__image_transform(crop),0))],0)
                elif self.manage_nan_values is 'avoid':
                    print('FileListProcessor_Semantic_Segmentation: crops with Nan values will be avoided')
                    def concat_only_no_nans_crop(batch_in, crop_candidate):
                        return tf.cond(tf.reduce_any(tf.is_nan(crop_candidate)),
                                            lambda : batch_in,
                                            lambda : tf.concat([batch_in, crop_candidate],0))
                    if self.balance_classes_distribution is True and self.no_reference is False:
                        ref_slice=tf.slice(crop,
                                            begin=[0,0,self.single_image_raw_depth],
                                            size=[-1,-1,self.single_image_reference_depth])

                        batch = tf.cond(tf.greater(get_sample_entropy(tf.reshape(ref_slice,[-1])), self.classes_entropy_threshold, name='minimum_labels_entropy_selection'),
                                                   lambda : concat_only_no_nans_crop(batch, tf.expand_dims(self.__image_transform(crop),0)),#if entropy is enough
                                                   lambda : batch)
                    else:
                        batch =  concat_only_no_nans_crop(batch, tf.expand_dims(self.__image_transform(crop),0))
                else:
                    print('FileListProcessor_Semantic_Segmentation: Nan values will throw error')
                    if self.balance_classes_distribution is True  and self.no_reference is False:
                        ref_slice=tf.slice(crop,
                                            begin=[0,0,self.single_image_raw_depth],
                                            size=[-1,-1,self.single_image_reference_depth])
                        batch = tf.cond(tf.greater(get_sample_entropy(tf.reshape(ref_slice,[-1])), self.classes_entropy_threshold, name='minimum_labels_entropy_selection'),
                                                   lambda : tf.concat([batch, tf.expand_dims(self.__image_transform(crop),0)],0),#if entropy is enough
                                                   lambda : batch)
                    else:
                        batch =  tf.concat([batch, tf.expand_dims(self.__image_transform(crop),0)],0)

                return index+1, input_image,batch,top_coord,left_coord,patch_size,nb_patch
            #stop after nb_patch crop
            def cond(index, input_image, batch, top_coord, left_coord, patch_size, nb_patch):
                return index<nb_patch

            idx, _temp1,crops, _temp2,_temp3,_temp4, _temp5 = tf.while_loop(cond=cond,body=body,loop_vars=[idx,input_image, raw_sample_tensor_4d, left_coord, top_coord, self.patchSize, self.nbPatches], shape_invariants=[tf.TensorShape(None), input_image.shape, tf.TensorShape([None, None, None, None]), tf.TensorShape(None), tf.TensorShape(None), tf.TensorShape(None),tf.TensorShape(None)], parallel_iterations=10, name="generating_crops")

            #enqeue being sure that no nan is included
            assert_op = tf.Assert(tf.logical_not(tf.reduce_any(tf.is_nan(crops))), [crops], name='crops_detect_nan_values')
            with tf.control_dependencies([assert_op]):
                self.generate_output_crop = self.deepnet_data_queue.enqueue_many(crops)


    def __image_transform(self, input_image):
        ''' apply a set of transformation to an input image
        @param input_image, the image to be transformed. It must be a stack of
        the raw image (first layers) followed by the reference layer(s)
        @return the transformed raw+reference concatenated image, only geometric transforms are applied to the reference image
        '''
        with tf.name_scope('image_transform'):
            #retreive a single crop
            """ standard cropping scheme """
            transformed_image=input_image
            #apply basic transforms
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
                single_image_channels = tf.image.per_image_standardization(single_image_channels)

            if self.no_reference is False:#get back to the input+reference images concat
                return tf.concat([tf.cast(single_image_channels, dtype=tf.float32), reference_img], axis=2)
            else:
                return tf.cast(single_image_channels, dtype=tf.float32)

    def start(self, session, coordinator):
        """ start the input pipeline and feed the deepnet_data_queue
        coordinator: the coordinator to be used to control involved threads
        """

        print('-------> starting data queue feeding threads')
        self.enqueue_threads = tf.train.start_queue_runners(sess=session,
                                                            coord=coordinator,
                                                            daemon=True,
                                                            start=True)

    def __class_balance_filter(self, samples_set):

        ''' method that selects a subset of the samples in order to maintain
        a more fair class distribution
        @param: the set of sample images which first layers is made of the rax pixel data
                while the last layer is the reference dense labels
        @return: the set of filtered samples
        '''
        with tf.name_scope('crops_selection_for_class_balancing'):
            #first extract the dense label layer for each sample
            dense_labels=tf.slice( samples_set,
                                            begin=[0,0,0,self.single_image_raw_depth],
                                            size=[-1,-1,-1,self.single_image_reference_depth])
            #count the number of class representatives
            #whole_samples_classes_counts=tf.unique_with_counts(tf.reshape(dense_labels, [-1]), out_idx=tf.int32, name="dense_labels_class_counts_all_samples")
            #classes_hist = tf.histogram_fixed_width(tf.reshape(dense_labels, [-1]), [0,self.], nbins=5)
            #tf.Print(dense_labels, [whole_samples_classes_counts], summarize=20, first_n=7)

            #get a the vector of selected samples, force reshape to facilitate shape checks at tf.boolean_mask level
            samples_entropies=tf.reshape(tf.py_func(get_samples_entropies, [dense_labels], tf.float32), [self.nbPatches])
            selection_mask=tf.greater(samples_entropies, self.classes_entropy_threshold, name='minimum_labels_entropy_selection')
            #select samples and return it
            return tf.boolean_mask(samples_set, selection_mask, name='samples_filter_for_class_distrib_balance')

    def create_input_sample_producer(self):
        ''' given the chosen mode (using pairs of raw+ref image or using a single raw+ref(lastchannel) image)
            -> create the first step of the input pipeline: a files filename queue producer and image sample loader
            @return the raw_image_sample (raw data plus the last channel as the semantic labels reference)
        '''
        raw_sample=None
        number_of_epoch=None

        if self.image_pairs_raw_ref_input:
            self.queue_filenames=tf.train.slice_input_producer([self.filelist_raw_data, self.filelist_reference_data],
                                        num_epochs=number_of_epoch,
                                        shuffle=self.shuffle_samples,
                                        seed=None,
                                        capacity=self.num_preprocess_threads,
                                        shared_name=None,
                                        name='input_filenames_queue')
            #read the related image and semantic labels image
            raw_sample=self.__load_raw_ref_images(raw_img_filename=self.queue_filenames[0], ref_img_filename=self.queue_filenames[1])
        else: #raw and ref data in the same image of only raw data use cases
            self.queue_filenames=tf.train.input_producer(self.filelist_raw_data,# self.filelist_raw_data],,
                                        num_epochs=number_of_epoch,
                                        shuffle=self.shuffle_samples,
                                        seed=None,
                                        capacity=self.num_preprocess_threads,
                                        shared_name=None,
                                        name='input_filenames_queue')
            #read the related image and semantic labels image
            raw_sample=self.__load_raw_ref_single_image(raw_img_filename=self.queue_filenames.dequeue())

        return raw_sample

    def __create_pipeline_transform_each_crop(self):
        """ input pipeline is defined on the CPU parameters
        """
        with tf.device("/cpu:0"),tf.name_scope(FileListProcessor_Semantic_Segmentation.dataprovider_namescope+'_gen_crops_then_transform'):
            """create a first "queue" (actually a list) of pairs of image filenames
            and generate data samples (whole read images)
            """
            self.raw_sample=self.create_input_sample_producer()

            """create a second queue that will be fed by QueueRunners that prepare the data
            from the retreived filenames
            """
            if self.ordered_full_frame_mode == True:
                self.cropSize=self.fullframe_ref_shape
            else:
                self.cropSize=[self.patchSize,self.patchSize,self.single_image_raw_depth+self.single_image_reference_depth]
            print('Deep net will be fed by samples of shape='+str(self.cropSize))
            self.deep_data_queue_capacity= self.batch_size_train*self.max_patches_per_image*self.num_preprocess_threads
            min_after_dequeue = (self.deep_data_queue_capacity*3)/4
            print('data queue values: min_after_dequeue={min_after_dequeue},data_queue_capacity={capacity}'.format(min_after_dequeue=min_after_dequeue,capacity=self.deep_data_queue_capacity))

            if self.shuffle_samples: #randomized queues
                self.deepnet_data_queue = tf.RandomShuffleQueue(capacity=self.deep_data_queue_capacity,
                                                 min_after_dequeue=min_after_dequeue,
                                                 dtypes='float',
                                                 shapes=self.cropSize,
                                                 name='prefetched_data_queue')
            else: #fifo queues
                self.deepnet_data_queue = tf.FIFOQueue(capacity=self.deep_data_queue_capacity,
                                           dtypes='float',
                                           shapes=self.cropSize,
                                           name='prefetched_data_queue')

            #monitor queues:
            #tf.summary.scalar('raw_crops_queue_size', self.raw_crops_queue.size())
            tf.summary.scalar('data_queue_size', self.deepnet_data_queue.size())
            """ define the graph of the queue runners
            """

            #print('concatenated raw data+ dense semantic labels, shape='+str(self.sample.get_shape().as_list()))
            """if reading one by one (1 thread) and preserving aspect ratio and using 1 crop per image
            -> then, enqueue the full image with full reference
            """
            if  self.ordered_full_frame_mode == True:
                with tf.name_scope('full_raw_frame_prefectching'):
                    if self.apply_whitening:     # Subtract off the mean and divide by the variance of the pixels.
                        self.raw_sample=self.__whiten_sample(self.raw_sample)

                    #enqueue whole raw data + reference into the queue
                    self.generate_output_crop=self.deepnet_data_queue.enqueue(self.raw_sample)
                return
            #in 'crop mode', first enqueue raw crops in a first queue
            self.__generate_crops(self.raw_sample)

        print('Input pipe graph is now defined, ready to be started')

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
        batch_size_train: optionnal but well suited to optimize queue lenght to not limit training speed
        use_opencv_imread: set False if data should be loaded from tensorflow image read methods (for now, jpeg and png only)
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
                    batch_size_train=50,
                    use_opencv_imread=False,
                    balance_classes_distribution=False,
                    classes_entropy_threshold=0.6,
                    opencv_read_flags=cv2.IMREAD_UNCHANGED, #cv2.IMREAD_LOAD_GDAL | cv2.IMREAD_ANYDEPTH ):
                    field_of_view=0,
                    manage_nan_values=None):
        self.filelist_raw_data=filelist_raw_data
        self.filelist_reference_data=filelist_reference_data
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
        self.batch_size_train=batch_size_train
        self.use_opencv_imread=use_opencv_imread
        self.balance_classes_distribution=balance_classes_distribution
        self.classes_entropy_threshold=classes_entropy_threshold
        self.opencv_read_flags=opencv_read_flags
        self.field_of_view = field_of_view
        self.manage_nan_values=manage_nan_values

        if self.image_area_coverage_factor<=0:
            raise ValueError('Error when constructing DataProvider: image_area_coverage_factor must be above 0')

        #first read the first raw and reference images to get aspect ratio and depth
        raw0=cv2.imread(filelist_raw_data[0],opencv_read_flags)
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
            print('read firt reference image {filepath} of shape {shape}'.format(filepath=filelist_reference_data[0], shape=ref0.shape))
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

        self.ordered_full_frame_mode=self.patch_ratio_vs_input==self.max_patches_per_image==1
        if self.ordered_full_frame_mode == True:
            print('==> each image will finally be processed globally following')
        else:
            print("Image patches will be of size:"+str(self.patchSize))
            print("Image patches ratio vs input is:"+str(self.patch_ratio_vs_input))

        # checking if field_of_view is odd
        if self.field_of_view%2 == 0 and self.field_of_view != 0:
            raise ValueError('field_of_view must be odd or 0 (current : {})'.format(self.field_of_view))

        #create the input pipeline
        self.__create_pipeline_transform_each_crop()

        print('prepare queue runners')
        self.threads_enqueue_output_crops = tf.train.QueueRunner(self.deepnet_data_queue,
                                                            [self.generate_output_crop] * self.num_preprocess_threads)
        tf.train.add_queue_runner(self.threads_enqueue_output_crops)
        #-> now, queue runner of this data pipeline only has to be started using a coordinator


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
    @returns a dense data tensor
    '''
    #preparing input data features, convert to the appropriate type
    data_features=[]
    for data_col in features_labels['data_cols']['names_opt_categories_or_buckets']:
       print('***preparing input data column:'+str(data_col))
       if len(data_col)==1:
           print('----->numeric data to be casted as float 32')
           data_features.append(tf.feature_column.numeric_column(key=data_col['name']))
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
    data_vectors = tf.feature_column.input_layer(data_dict, data_features)
    print('*** data_vectors='+str(data_vectors))

    return data_vectors


def FileListProcessor_csv_time_series(files, csv_field_delim, record_defaults_values, nblines_per_block, queue_capacity, shuffle_batches, device="/cpu:0"):
    ''' define a queue that prefetches 1D vectors coming from a set of csv files
        records that can have empty cells but default values must be specified to replace empty spaces
        @param files: the list of input csv files to read from
        @param csv_field_delim: the cell delimiter in the csv file
        @param record_defaults_values: the list of default values, for example: [['default_txt'],[1],[1.0]]
        @param nblines_per_block:the number of lines to read at a time : this will enable to dequeue this number lines as a single datablock
        @param queue_capacity: the number of sample slots available in the queue
        @param device: the device where to place the data pipeline
    '''
    with tf.device(device),tf.name_scope('csv_file_line_blocks_read_enqueue'): # force input pipeline to be ran on the main cpu
        filename_queue = tf.train.string_input_producer(files)
        reader = tf.TextLineReader(skip_header_lines=1)
        key, value = reader.read_up_to(filename_queue, num_records=nblines_per_block, name='read_lines')
        # Default values, in case of empty columns. Also specifies the type of the
        # decoded result.
        features = tf.decode_csv(
            value, record_defaults=record_defaults_values, field_delim=csv_field_delim) #keep all the the first column (timestamp)
        #stack all but first column in a single tensor
        timestamps=tf.cast(features[0], dtype=tf.string)
        raw_data_sample=tf.cast(tf.stack(features[1:], axis=1), dtype=tf.float32)
        '''raw_data_sample = tf.Print(raw_data_sample, [raw_data_sample],
               'read lines = ', summarize=2, first_n=10)'''
        print('timestamps='+str(timestamps))
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
        conditionned_enqueue_op=tf.cond(check_nblines_per_block_OK, lambda: data_queue.enqueue([timestamps,raw_data_sample]), lambda:tf.no_op())

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

def FileListProcessor_csv_lines(files, csv_field_delim, queue_capacity, shuffle_batches, batch_size, features_labels, device="/cpu:0", debug=False):
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

        @param device: the device where to place the data pipeline
    '''

    with tf.device(device),tf.name_scope('csv_file_lines_read_enqueue'): # force input pipeline to be ran on the main cpu
        def decode_csv(line):
            if debug is True:
                line=tf.Print(line,[line], message='RAW csv line=')
            parsed_line = tf.decode_csv(line, record_defaults=features_labels['all_cols']['record_defaults'], field_delim=csv_field_delim)
            features = dict(zip(features_labels['all_cols']['names'], parsed_line))
            unused_cols = set(features_labels['all_cols']['names']) - {col['name'] for col in features_labels['data_cols']['names_opt_categories_or_buckets']} \
                                                                    - {col for col in features_labels['labels_cols']['names']}
            print('### UNUSED COLUMNS='+str(unused_cols))
            # Remove unused columns
            for col in unused_cols:
                features.pop(col)
            print('### USED COLUMNS (data and labels)='+str(features))
            return features

        dataset = (tf.data.TextLineDataset(files)  # Read text file
           .skip(1)  # Skip header row
           .map(decode_csv, num_parallel_calls=4)  # Decode each line in a multi thread mode (asynchronous reading)
           .cache() # Warning: Caches entire dataset, can cause out of memory
           .repeat(None)    # Repeats dataset this # times
           .batch(batch_size)
           .prefetch(5*batch_size)  # Make sure you always have 1 batch ready to serve
        )
        if shuffle_batches is True:
            dataset=dataset.shuffle(buffer_size=5*batch_size)

        return dataset

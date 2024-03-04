import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import plot_model

def basis_identity(theta, ts_length, name):
	""" basic identity basis: thetas are directly considered as for/back casts
	=> no additionnal computation
	"""
	assert theta.get_shape().as_list()[-1]==ts_length
	return theta

def basis_func_trend(theta, ts_length, name):
	"""
	setup a polynomial trend parametrized by the incoming theta features
	-> applied to a single time series, either forecast of backcast
	Args:
		theta: the parameters that control the trend function, its last dimension (-1) sets the polynom degree
		ts_lengh: the timeseries length
	"""
	# the trend polynomial degree is directly related to the dimensions of theta
	polynomial_degree_plus1=theta.get_shape().as_list()[-1]
	polynomial_degree = polynomial_degree_plus1-1
	print('trend basis_func_trend basis, degree=', polynomial_degree)
	timesteps=np.arange(ts_length, dtype=np.float)/ts_length
	powers=np.arange(polynomial_degree_plus1)
	T=np.array([timesteps**p for p in powers], dtype=np.float32)
	
	""" finally apply theta parameters as coefficients on the target polynom
	this is actually a simple matrix product (T*theta) that can be done by a not trainable
	dense neural layer  
	...you may see some equivalent calculus with other codes such as:
	einsum_np=np.einsum('bp,pt->bt', theta.numpy(), T.numpy())
	dot_np=np.dot(theta.numpy(), T.numpy())
	einsum_tf=tf.tensordot(theta, T, axes=1)
	"""
	basis_tf=tf.keras.layers.Dense(units=ts_length,
									kernel_initializer=tf.keras.initializers.Constant(value=T),
									trainable=False,
									use_bias=False,
									name=name+'trend_polynom_'+str(polynomial_degree))(theta)
	return basis_tf

def test_basis_func_trend():
	# demo/test code using the basis_func_trend function:	
	thetas=tf.constant(np.random.normal(size=(3, 10)))
	#thetas=tf.constant([[0, 0.1, 0.2], [-1, -10.1, -1.3]])
	res=basis_func_trend(theta=thetas, ts_length=10, name='test_trend')
	print('res=', res)
	import matplotlib.pyplot as plt
	plt.plot(res.numpy().transpose())
	plt.title('trend samples')
	plt.show()
#test_basis_func_trend()

def basis_func_seasonality(theta, ts_length, name):
	"""
	setup a seasonal model parametrized by the incoming theta features
	-> applied to a single time series, either forecast of backcast
	Args:
		theta: the parameters that control the trend function, its last dimension (-1) sets twice the number of harmonics
		ts_lengh: the timeseries length
	"""
	p = theta.get_shape().as_list()[-1]
	p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)
	timesteps=np.arange(ts_length, dtype=np.float)/ts_length
	s1 = np.array([np.cos(2 * np.pi * i * timesteps) for i in range(p1)])
	s2 = np.array([np.sin(2 * np.pi * i * timesteps) for i in range(p2)])
	if p == 1:
		T = s2
	else:
		T = np.concatenate([s1, s2], axis=0).astype(np.float32)
	basis_tf=tf.keras.layers.Dense(units=ts_length,
									kernel_initializer=tf.keras.initializers.Constant(value=T),
									trainable=False,
									use_bias=False,
									name=name+'season_harmonics'+str(p))(theta)
	return basis_tf

def test_basis_func_seasonality():
	# demo/test code using the basis_func_trend function:	
	thetas=tf.constant(np.random.normal(size=(3, 10)))
	#thetas=tf.constant([[0, 0.1, 0.2, 1, 1.1, 1.2], [-1, -10.1, -1.3, -2.3, -5.3, 4.0]])
	res=basis_func_seasonality(thetas, ts_length=100, name='test_seasonality')
	import matplotlib.pyplot as plt
	plt.plot(res.numpy().transpose())
	plt.title('seasonality samples')
	plt.show()
#test_basis_func_seasonality()

def static_features_encoder(features, neurons, dropout_rate=0.5, activation=tf.keras.layers.ReLU(), name_suffix:str=''):
	""" basic feature encoding with a single non linear layer"""
	enc= tf.keras.layers.Dense(units=neurons,
								kernel_initializer=tf.keras.initializers.Orthogonal(),
								kernel_regularizer=tf.keras.regularizers.L2(0.0001),
								activation="linear",
								name="feature_enc_"+name_suffix)
	if activation == 'crelu':
		enc=tf.keras.layers.Concatenate(axis=-1)([tf.keras.layers.ReLU()(enc),
												tf.keras.layers.ReLU()(-enc)
												])
	else:
		enc=activation(enc)

	if dropout_rate>0:
		enc = tf.keras.layers.Dropout(rate=dropout_rate)(enc)
	return enc

def nBeats_block(inputs, theta_size:int, backcast_horizon:int, forecast_horizon:int, n_layers:int=4, n_neurons:int=512, basis_func=basis_identity, dropout_rate=0.5, name:str=None):
	x=inputs
	for id in range(n_layers):
		x=tf.keras.layers.Dense(n_neurons,
							kernel_initializer=tf.keras.initializers.Orthogonal(),
							kernel_regularizer=tf.keras.regularizers.L2(0.0001),
							activation="relu")(x)
		#if dropout_rate>0:
		#	x = tf.keras.layers.Dropout(rate=dropout_rate)(x) 
	#finally predict horizon and backward transformation
	thetas = tf.keras.layers.Dense(units=2*theta_size,
							kernel_initializer=tf.keras.initializers.VarianceScaling(),
							kernel_regularizer=tf.keras.regularizers.L2(0.0001),
							activation="linear",
							name="thetas_"+name)(x)
	
	theta_backward=thetas[:,:theta_size]
	theta_forward=thetas[:,theta_size:]
	return basis_func(theta_backward, backcast_horizon, name=name+'_backcast'), basis_func(theta_forward, forecast_horizon, name=name+'_forcast')

def model(usersettings):

	# Create input layer (the initial residual)
	model_input= tf.keras.layers.Input(shape=(usersettings.hparams['tsLengthIn']),
										name="ts_input")

	# 3. define the nbeats model
	residuals =model_input
	model_outputs={}
	for stack_type in ('TREND', 'SEASON'):
		with tf.name_scope('nBeats_'+stack_type):
			for id in range(usersettings.hparams['nstacks']):
				block_name=stack_type+str(id)
				with tf.name_scope('stack_'+str(id)):
					#choose among the available basis functions
					basis_func=None
					if stack_type == 'TREND':
						basis_func=basis_func_trend
					elif stack_type == 'SEASON':
						basis_func=basis_func_seasonality
					#setup a single block with the current stack
					backcast, forecast = nBeats_block(residuals,
														theta_size=usersettings.hparams['nneurons'],
														backcast_horizon=usersettings.hparams['tsLengthIn'],
														forecast_horizon=usersettings.hparams['tsLengthOut'],
														n_layers=usersettings.hparams['nlayers'],
														n_neurons=usersettings.hparams['nneurons'],
														basis_func=basis_func,
														name=block_name)
					globalcast=tf.keras.layers.Concatenate(axis=-1)([backcast, forecast])
					model_outputs[block_name]=tf.keras.layers.Activation('linear', dtype='float32', name=block_name)(globalcast)
					
					# update residuals and forecast
					residuals = tf.keras.layers.subtract([residuals, backcast], name="subtract_"+block_name)
					if 'global_forecast' in locals():
						global_forecast = tf.keras.layers.add([global_forecast, forecast], name="add_f_"+block_name)
						global_backcast = tf.keras.layers.add([global_backcast, backcast], name="add_b_"+block_name)
					else:
						global_forecast = forecast
						global_backcast = backcast
	# declare final main output
	model_outputs['forecast']=tf.keras.layers.Activation('linear', dtype='float32', name='forecast')(global_forecast)
	model_outputs['backcast']=tf.keras.layers.Activation('linear', dtype='float32', name='backcast')(global_backcast)
	# 7. Put the stack model together

	model = tf.keras.Model(inputs=model_input, 
		                 outputs=model_outputs, 
		                 name="model_N-BEATS")
	return model

if __name__ == "__main__":
	import matplotlib.pyplot as plt

	# M"/M4 model hyperparameters
	# Values from N-BEATS paper Figure 1 and Table 18/Appendix D
	N_EPOCHS = 5000 # called "Iterations" in Table 18

	#create a hyperparameters object to mimic the functionning of the framework in this test function
	class Setup(object):
		def __init__(self):
			self.hparams={}
			self.hparams['batch_size']=64
			self.hparams['tsLengthIn']=100
			self.hparams['tsLengthOut']=50
			self.hparams['nneurons']=48
			self.hparams['nlayers']=2
			self.hparams['nstacks']=2
	config=Setup()


	#create a fake dataset
	x=np.arange(start=-1000, stop=1000)
	y=(x/100+np.sin(x/10)).astype(np.float32)
	import pandas as pd
	df=pd.read_csv('/home/alben/workspace/Datasets/Trading/BTcoin/market-price.csv')
	x=df['Timestamp'].to_numpy()
	y=df['market-price'].to_numpy().astype(np.float32)
	
	#y+=np.random.normal(size=y.shape)
	"""plt.plot(x,y)
	plt.title('whole dataset')
	plt.show()
	"""
	#setup tensorflow config
	tf.random.set_seed(42)

	#create the related Tensorflow dataset
	temporal_series_length=config.hparams['tsLengthIn']+config.hparams['tsLengthOut']
	test_ratio=0.2
	test_cut_index=int(len(x)*(1-test_ratio))
	print('temporal_series_length', temporal_series_length)
	def create_windows_dataset(y, isTrainingData):
		dataset=tf.data.Dataset.from_tensor_slices(y)
		# extract windows and set them as samples 
		dataset=dataset.window(size=temporal_series_length, shift=1, drop_remainder=True).flat_map(lambda x:x.batch(temporal_series_length))
		print('windows', dataset)
		def split_past_future(sample):
			print('sample', sample)
			return sample[:config.hparams['tsLengthIn']], {'forecast':sample[config.hparams['tsLengthIn']:], 'backcast':sample[:config.hparams['tsLengthIn']]}
		dataset=dataset.map(split_past_future)
		if isTrainingData:
			dataset=dataset.shuffle(500)
		#finalize pipeline with batching and prefetching
		dataset=dataset.batch(config.hparams['batch_size'], drop_remainder=True).prefetch(1)
		return dataset

	dataset_train=create_windows_dataset(y[:test_cut_index], isTrainingData=True)
	dataset_val  =create_windows_dataset(y[test_cut_index:], isTrainingData=False)
	
	#just to see the ordered val dataset values:
	
	"""
	for sample in dataset_val:
		print(sample)
		print('back', sample[0].shape)
		print('for', sample[1]['forecast'].shape)
	"""	
	# Basic tests    
	test_model = model(Setup())
	test_model.summary()
	# 8. Compile with MAE loss and Adam optimizer
	def residuals_loss(y_true,y_pred):
		print('ae y_true=',y_true)
		print('ae y_pred=',y_pred)
		return tf.reduce_sum(tf.math.abs(y_true-y_pred), name='residual_loss')#tf.keras.losses.MAE(y_pred, tf.zeros_like(y_pred))#since y_pred is already output-input
		
	test_model.compile(loss={'forecast':tf.keras.losses.MeanAbsoluteError(), 'backcast':tf.keras.losses.MeanAbsoluteError()},
		        optimizer=tf.keras.optimizers.Adam(0.001),
		        metrics=["mae", "mse", "mape"])
	try:
		plot_model(test_model, to_file='nBeats.png', show_shapes=True)
	except Exception as e:
		print('could not plot model, error=', e)
	# 9. Fit the model with EarlyStopping and ReduceLROnPlateau callbacks
	test_model.fit(dataset_train,
		    epochs=N_EPOCHS,
		    validation_data=dataset_val,
		    verbose=True, # prevent large amounts of training outputs
		    # callbacks=[create_model_checkpoint(model_name=stack_model.name)] # saving model every epoch consumes far too much time
		    callbacks=[tf.keras.callbacks.EarlyStopping(monitor="forecast_loss", patience=200, restore_best_weights=True),
		              tf.keras.callbacks.ReduceLROnPlateau(monitor="forecast_loss", patience=100, verbose=1)])
	
	x=np.arange(temporal_series_length)
	x_backcast=x[:config.hparams['tsLengthIn']]
	x_forecast=x[config.hparams['tsLengthIn']:]
	display_samples=5
	for test_sample in dataset_val:
		ts_backcast=test_sample[0].numpy()[:display_samples,:]
		ts_future=test_sample[1]['forecast'].numpy()[:display_samples,:]
		preds=test_model.predict(ts_backcast)
		plt.plot(x_forecast, preds['forecast'][:display_samples,:].transpose(), ('r'))
		plt.plot(x_backcast, preds['backcast'][:display_samples,:].transpose(), ('r'))
		"""
		for pred_name in preds.keys():
			if pred_name=='forecast':
				plt.plot(x_forecast, preds['forecast'][:display_samples,:].transpose(), ('r'))
			else:
				plt.plot(x_backcast, preds[pred_name][:display_samples,:].transpose(), ('r'))
		"""		
		#display backcast and forecast reference values
		plt.plot(x_backcast, ts_backcast.transpose())
		plt.plot(x_forecast, ts_future.transpose(), ('b'))
		plt.show()
		
	test_model.save('nBeats-save.h5')

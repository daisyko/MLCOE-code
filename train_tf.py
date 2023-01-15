import os
import argparse
import json
import numpy as np 
import tensorflow  as tf
from tensorflow.keras.callbacks import ModelCheckpoint

from utils.util_tf import print_size, training_loss, calc_diffusion_hyperparams
from utils.util_tf import get_mask_mnr, get_mask_bm, get_mask_rm


from imputers.DiffWaveImputer_tf import DiffWaveImputer
#from imputers.SSSDSAImputer import SSSDSAImputer
from imputers.SSSDS4Imputer_tf import SSSDS4Imputer

def train(output_directory,
	ckpt_iter,
	n_iters,
	iters_per_ckpt,
	iters_per_logging,
	learning_rate,
	use_model,
	only_generate_missing,
	masking,
	missing_k):
	#Train Diffusion Models
	"""
	Param:
	output_directory(str): save model to
	ckpt_iter (int or "max"): the pretrained checkpoint to be loaded; the max iteration if "max" 
	data_path (str): path to dataset, numpy array
	n_iters (int): # of iterations to train
	iters_per_ckpt (int): # of iteration to save checkpoint; default = 10k
	iters_per_logging (int): # of iterations to save training log and compute validation loss; default = 100
	learning_rate (float)
	use_model (int): 0=DiffWave; 1=SSSDSA; 2=SSSDS4
	only_generate_missing (int): 0=all sample diffusion; 1=only apply diffusion to missing portions of the signal
	masking (str): 'mnr'=missing not at random; 'bm'=blackout missing; 'rm'=random missing
	missing_k (int): k missing time steps for each feature across the sample length
	"""

	#generate local path for experinmennt
	local_path =  "T{}_beta0{}_betaT{}".format(diffusion_config["T"],diffusion_config["beta_0"],diffusion_config["beta_T"])

	#make output dir
	output_directory = os.path.join(output_directory, local_path)
	os.makedirs(output_directory, exist_ok=True)
	os.chmod(output_directory, 0o775)
	print("output directory", output_directory, flush=True)

	#predefine model
	if use_model == 0:
		net = DiffWaveImputer(**model_config)
	elif use_model == 1:
		net = SSSDSAImputer(**model_config)
	elif use_model == 2:
		net = SSSDS4Imputer(**model_config)
	else:
		print('Model chosen not available.')

	#define optimizer
	optimizer =  tf.keras.optimizers.Adam(learning_rate=learning_rate) 

	ckpt = tf.train.Checkpoint(optimizer=optimizer, net=net)
	manager = tf.train.CheckpointManager(ckpt, output_directory, max_to_keep=3)
	#load checkpoint
	if ckpt_iter == "max":
		ckpt.restore(manager.latest_checkpoint)
		if manager.latest_checkpoint:
			print ('Successfullly loaded model from {}'.format(manager.latest_checkpoint))
			#need to get ckpt_iter value
		else:
			print('No valid checkpoint model found, start training from initialization try.')
			ckpt_iter = -1
	else:
		ckpt_iter = -1
		print('No valid checkpoint model found, start training from initialization.')


	### Custom data loading and reshaping ###
	training_data = np.load(trainset_config['train_data_path'])
	training_data = np.split(training_data, 160, 0)
	training_data = np.array(training_data)

	training_data = tf.convert_to_tensor(training_data,dtype = tf.float32)
	print('Data loaded')

	print_size(net)

	# training
	#n_iter = ckpt_iter + 1 ####need to uncommand after getting ckpt_iter
	n_iter = 1
	while n_iter < n_iters + 1:
		for batch in training_data:

			if masking == "rm":
				transposed_mask = get_mask_rm(batch[0], missing_k)
			elif masking == 'mnr':
				transposed_mask = get_mask_mnr(batch[0], missing_k)
			elif masking == 'bm':
				transposed_mask = get_mask_bm(batch[0], missing_k) #in util

			mask = tf.transpose(transposed_mask, perm=[1,0])
			
			mask = tf.expand_dims(mask, 0) #expand dimention in 0
			to_repeat = tf.constant([batch.shape[0], 1, 1], tf.int32) #the size to repeat
			mask = tf.tile(mask, to_repeat) #the final mask

			loss_mask = ~mask.bool()
			batch = tf.transpose(batch, perm=[0,2,1]) #may not need to transpose in tf

			assert batch.shape == mask.shape == loss_mask.shape


			X = batch, batch, mask, loss_mask
			with tf.GradientTape() as tape:
				loss = training_loss(net, tf.keras.losses.MeanSquaredError(), X, diffusion_hyperparams,
								only_generate_missing=only_generate_missing)

			grads = tape.gradient(loss, net.trainable_variables)
			optimizer.apply_gradients(zip(grads, net.trainable_variables))				

			if n_iter % iters_per_logging == 0:
				print("iteration: {} \tloss: {}".format(n_iter, loss.numpy()))

			# save checkpoint
			if n_iter > 0 and n_iter % iters_per_ckpt == 0:
				save_path = manager.save()
				print('model at iteration %s is saved' % n_iter)

			n_iter += 1


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--config', type=str, default='config/SSSDS4.json',
						help='JSON file for configuration')

	args = parser.parse_args()

	#read and load preset values from config file
	with open(args.config) as f:
		data = f.read()

	config = json.loads(data)
	print(config)

	train_config = config["train_config"]  # training parameters

	global trainset_config
	trainset_config = config["trainset_config"]  # to load trainset

	global diffusion_config
	diffusion_config = config["diffusion_config"]  # basic hyperparameters

	global diffusion_hyperparams
	diffusion_hyperparams = calc_diffusion_hyperparams(
		**diffusion_config)  # dictionary of all diffusion hyperparameters

	global model_config
	if train_config['use_model'] == 0:
		model_config = config['wavenet_config']
	elif train_config['use_model'] == 1:
		model_config = config['sashimi_config']
	elif train_config['use_model'] == 2:
		model_config = config['wavenet_config']

	train(**train_config)




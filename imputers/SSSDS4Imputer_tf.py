import math
from utils.util_tf import calc_diffusion_step_embedding
#from imputers.S4Model_tf import S4Layer
from imputers.S4Model import S4Layer
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_addons as tfa



def swish(x):
	#to find element wise sigmoid of x
	return x * tf.math.sigmoid(x)


class Conv(tf.keras.Model):
	def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
		super(Conv, self).__init__()
		initializer = tf.keras.initializers.HeNormal()
		self.conv = tf.keras.layers.Conv1D(out_channels, kernel_size, dilation_rate=dilation, padding='same',kernel_initializer=initializer)
		self.conv = tfa.layers.WeightNormalization(self.conv)

	def call(self, x):
		out = self.conv(x)
		return out


	
	
class ZeroConv1d(tf.keras.Model):
	def __init__(self, in_channel, out_channel):
		super(ZeroConv1d, self).__init__()
		self.conv = tf.keras.layers.Conv1D(out_channel, kernel_size=1, padding='valid', kernel_initializer=tf.keras.initializers.Zeros(),bias_initializer='zeros')
		self.conv.build((None,None,in_channel))

	def call(self, x):
		out = self.conv(x)
		return out



class Residual_block(tf.keras.Model):
	def __init__(self, res_channels, skip_channels, 
				 diffusion_step_embed_dim_out, in_channels,
				s4_lmax,
				s4_d_state,
				s4_dropout,
				s4_bidirectional,
				s4_layernorm):
		super(Residual_block, self).__init__()
		self.res_channels = res_channels


		self.fc_t = tf.keras.layers.Dense(self.res_channels, input_shape=(diffusion_step_embed_dim_out,), activation=None)
		self.fc_t.build((diffusion_step_embed_dim_out,))

		self.S41 = S4Layer(features=2*self.res_channels, 
						  lmax=s4_lmax,
						  N=s4_d_state,
						  dropout=s4_dropout,
						  bidirectional=s4_bidirectional,
						  layer_norm=s4_layernorm)
 
		self.conv_layer = Conv(self.res_channels, 2 * self.res_channels, kernel_size=3)

		self.S42 = S4Layer(features=2*self.res_channels, 
						  lmax=s4_lmax,
						  N=s4_d_state,
						  dropout=s4_dropout,
						  bidirectional=s4_bidirectional,
						  layer_norm=s4_layernorm)
		
		self.cond_conv = Conv(2*in_channels, 2*self.res_channels, kernel_size=1)  

		initializer_res = tf.keras.initializers.HeNormal()
		self.res_conv = tf.keras.layers.Conv1D(res_channels, 1, kernel_initializer=initializer_res) #,input_shape=(res_channels,)
		self.res_conv = tfa.layers.WeightNormalization(self.res_conv)
		self.res_conv.build((None,None,res_channels))

		initializer_skip = tf.keras.initializers.HeNormal()
		self.skip_conv = tf.keras.layers.Conv1D(skip_channels, 1, kernel_initializer=initializer_skip,input_shape=(res_channels,))
		self.skip_conv = tfa.layers.WeightNormalization(self.skip_conv)
		self.skip_conv.build((None,None,res_channels))

	def call(self, input_data):
		x, cond, diffusion_step_embed = input_data
		h = x
		B, C, L = x.shape
		assert C == self.res_channels                      
				 
		part_t = self.fc_t(diffusion_step_embed)
		part_t = part_t.view([B, self.res_channels, 1])  
		h = h + part_t
		
		h = self.conv_layer(h)
		h = self.S41(h.permute(2,0,1)).permute(1,2,0)     
		
		assert cond is not None
		cond = self.cond_conv(cond)
		h += cond
		
		h = self.S42(h.permute(2,0,1)).permute(1,2,0)
		
		out = tf.math.tanh(h[:,:self.res_channels,:]) * tf.math.sigmoid(h[:,self.res_channels:,:])

		res = self.res_conv(out)
		assert x.shape == res.shape
		skip = self.skip_conv(out)

		return (x + res) * math.sqrt(0.5), skip  # normalize for training stability


class Residual_group(tf.keras.Model):
	def __init__(self, res_channels, skip_channels, num_res_layers, 
				 diffusion_step_embed_dim_in, 
				 diffusion_step_embed_dim_mid,
				 diffusion_step_embed_dim_out,
				 in_channels,
				 s4_lmax,
				 s4_d_state,
				 s4_dropout,
				 s4_bidirectional,
				 s4_layernorm):
		super(Residual_group, self).__init__()
		self.num_res_layers = num_res_layers
		self.diffusion_step_embed_dim_in = diffusion_step_embed_dim_in

		self.fc_t1 = tf.keras.layers.Dense(diffusion_step_embed_dim_mid, input_shape=(diffusion_step_embed_dim_in,), activation=None)
		self.fc_t1.build((diffusion_step_embed_dim_in,))
		self.fc_t2 = tf.keras.layers.Dense(diffusion_step_embed_dim_out, input_shape=(diffusion_step_embed_dim_mid,), activation=None)
		self.fc_t2.build((diffusion_step_embed_dim_mid,))

		self.residual_blocks = [Residual_block(res_channels, skip_channels, 
													   diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
													   in_channels=in_channels,
													   s4_lmax=s4_lmax,
													   s4_d_state=s4_d_state,
													   s4_dropout=s4_dropout,
													   s4_bidirectional=s4_bidirectional,
													   s4_layernorm=s4_layernorm) for n in range(self.num_res_layers)]


			
	def call(self, input_data):
		noise, conditional, diffusion_steps = input_data

		diffusion_step_embed = calc_diffusion_step_embedding(diffusion_steps, self.diffusion_step_embed_dim_in)
		diffusion_step_embed = swish(self.fc_t1(diffusion_step_embed))
		diffusion_step_embed = swish(self.fc_t2(diffusion_step_embed))

		h = noise
		skip = 0
		for n in range(self.num_res_layers):
			h, skip_n = self.residual_blocks[n]((h, conditional, diffusion_step_embed))  
			skip += skip_n  

		return skip * math.sqrt(1.0 / self.num_res_layers)  


class SSSDS4Imputer(tf.keras.Model):
	def __init__(self, in_channels, res_channels, skip_channels, out_channels, 
				 num_res_layers,
				 diffusion_step_embed_dim_in, 
				 diffusion_step_embed_dim_mid,
				 diffusion_step_embed_dim_out,
				 s4_lmax,
				 s4_d_state,
				 s4_dropout,
				 s4_bidirectional,
				 s4_layernorm):
		super(SSSDS4Imputer, self).__init__()

		self.init_conv = tf.keras.Sequential()
		self.init_conv.add(tf.keras.Input(shape=(None,None,in_channels)))
		self.init_conv.add(Conv(in_channels, res_channels, kernel_size=1))
		self.init_conv.add(tf.keras.layers.ReLU())
		
		self.residual_layer = Residual_group(res_channels=res_channels, 
											 skip_channels=skip_channels, 
											 num_res_layers=num_res_layers, 
											 diffusion_step_embed_dim_in=diffusion_step_embed_dim_in,
											 diffusion_step_embed_dim_mid=diffusion_step_embed_dim_mid,
											 diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
											 in_channels=in_channels,
											 s4_lmax=s4_lmax,
											 s4_d_state=s4_d_state,
											 s4_dropout=s4_dropout,
											 s4_bidirectional=s4_bidirectional,
											 s4_layernorm=s4_layernorm)
		#self.residual_layer.build((int(diffusion_step_embed_dim_in),None,None))
		
		self.final_conv = tf.keras.Sequential()
		self.final_conv.add(tf.keras.Input(shape=(None,None,skip_channels)))
		self.final_conv.add(Conv(skip_channels, skip_channels, kernel_size=1))
		self.final_conv.add(tf.keras.layers.ReLU())
		self.final_conv.add(ZeroConv1d(skip_channels, out_channels))


	def call(self, input_data):
		noise, conditional, mask, diffusion_steps = input_data 

		conditional = conditional * mask
		conditional = tf.concat([conditional, mask.float()], 1)

		x = noise
		x = self.init_conv(x)
		x = self.residual_layer((x, conditional, diffusion_steps))
		y = self.final_conv(x)

		return y

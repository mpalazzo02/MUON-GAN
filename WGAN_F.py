import glob
import math
import os
import random
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib.colors import LogNorm
from scipy.linalg import sqrtm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, auc, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (Activation, BatchNormalization, Concatenate, Dense, Dropout, Flatten, Input,
                                      Lambda, LeakyReLU, Reshape)
from tensorflow.keras.models import Model
import tensorflow as tf
from xgboost import XGBClassifier
import matplotlib as mpl

mpl.use('Agg')  # Use the Agg backend for generating plots without a display environment


colours_raw_root = [[250,242,108],
					[249,225,104],
					[247,206,99],
					[239,194,94],
					[222,188,95],
					[206,183,103],
					[181,184,111],
					[157,185,120],
					[131,184,132],
					[108,181,146],
					[105,179,163],
					[97,173,176],
					[90,166,191],
					[81,158,200],
					[69,146,202],
					[56,133,207],
					[40,121,209],
					[27,110,212],
					[25,94,197],
					[34,73,162]]

colours_raw_root = np.flip(np.divide(colours_raw_root,256.),axis=0)
cmp_root = mpl.colors.ListedColormap(colours_raw_root)


def split_tensor(index, x):
	return Lambda(lambda x : x[:,:,index])(x)

print(tf.__version__)

processID = 5
os.environ["CUDA_VISIBLE_DEVICES"] = "%d"%processID

batch_size = 50
save_interval = 250
saving_directory = 'output/'

G_architecture = [100,100]
D_architecture = [100,100]

list_of_training_files = glob.glob('SPLIT_*.npy')

print(list_of_training_files)


print(' ')
print('Initializing networks...')
print(' ')


gen_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
disc_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)


def discriminator_loss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss

def generator_loss(fake_img):
    return -tf.reduce_mean(fake_img)

kernel_initializer_choice='random_uniform'
bias_initializer_choice='random_uniform'

##############################################################################################################
# Build Generative model ...
input_noise = Input(shape=(1,100))

H = Dense(int(G_architecture[0]))(input_noise)
#H = LeakyReLU(alpha=0.2)(H)
H = BatchNormalization(momentum=0.8)(H)

for layer in G_architecture[1:]:

	H = Dense(int(layer))(H)
	#H = LeakyReLU(alpha=0.2)(H)
	H = BatchNormalization(momentum=0.8)(H)

H = Dense(6,activation='tanh')(H)

generator = Model(inputs=[input_noise], outputs=[H])
generator.summary()
##############################################################################################################
def preprocess_data(X):
    for index in range(2, 8):
        X[:, index] = (X[:, index] - np.amin(X[:, index]))
        X[:, index] = (X[:, index] / np.amax(X[:, index]))
        X[:, index] = (X[:, index] * 2.) - 1.
    return X[:, 2:]
def load_and_preprocess_for_evaluation(batch_size):
    file_index = np.random.randint(0, len(list_of_training_files))
    X = np.load(list_of_training_files[file_index])
    X = preprocess_data(X)
    indices = np.random.permutation(len(X))[:batch_size]
    return X[indices]

##############################################################################################################
# Build Discriminator model ...
d_input = Input(shape=(1,6))

H = Flatten()(d_input)

for layer in D_architecture:
	H = Dense(int(layer))(H)
	H = LeakyReLU(alpha=0.2)(H)
	H = Dropout(0.2)(H)
	

d_output = Dense(1, activation='linear')(H)

discriminator = Model(d_input, [d_output])
discriminator.summary()
##############################################################################################################

def calculate_fid(features_real, features_generated):
    # Calculate mean and covariance statistics
    mu_real, sigma_real = features_real.mean(axis=0), np.cov(features_real, rowvar=False)
    mu_gen, sigma_gen = features_generated.mean(axis=0), np.cov(features_generated, rowvar=False)

    # Calculate mean and covariance difference
    ssdiff = np.sum((mu_real - mu_gen)**2.0)
    covmean = sqrtm(sigma_real.dot(sigma_gen))

    # Check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # Calculate FID
    fid = ssdiff + np.trace(sigma_real + sigma_gen - 2.0 * covmean)
    return fid
def gradient_penalty(batch_size, real_images, fake_images):
    # Get the interpolated image
    alpha = tf.random.normal([batch_size, 1, 6], 0.0, 1.0)
    diff = fake_images - real_images
    interpolated = real_images + alpha * diff

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred = discriminator(interpolated, training=True)

    grads = gp_tape.gradient(pred, [interpolated])[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp

# https://blog.paperspace.com/wgans/
d_steps = 5 # number of steps to train D for every one generator step
gp_weight = 10.
@tf.function
def train_step(real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]

        batch_size = tf.shape(real_images)[0]

        for i in range(d_steps):
            # Get the latent vector
            random_latent_vectors = tf.random.normal((batch_size, 1, 100), 0, 1)
            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                fake_images = generator(random_latent_vectors, training=True)
                # Get the logits for the fake images
                fake_logits = discriminator(fake_images, training=True)
                # Get the logits for the real images
                real_logits = discriminator(real_images, training=True)

                # Calculate the discriminator loss using the fake and real image logits
                d_cost = discriminator_loss(real_img=real_logits, fake_img=fake_logits)
                # Calculate the gradient penalty
                gp = gradient_penalty(batch_size, real_images, fake_images)
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost #+ gp * gp_weight

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            disc_optimizer.apply_gradients(zip(d_gradient, discriminator.trainable_variables))

        # Train the generator
        random_latent_vectors = tf.random.normal((batch_size, 1, 100), 0, 1)
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = generator(random_latent_vectors, training=True)
            # Get the discriminator logits for fake images
            gen_img_logits = discriminator(generated_images, training=True)
            # Calculate the generator loss
            g_loss = generator_loss(gen_img_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        gen_optimizer.apply_gradients(zip(gen_gradient, generator.trainable_variables))
        return g_loss, d_loss







fid_scores = []  #Empty list to store FID scores

start = time.time()

iteration = -1

loss_list = np.empty((0,3))

axis_titles = ['StartX', 'StartY', 'StartZ', 'Px', 'Py', 'Pz']

training_time = 0

t0 = time.time()

random.shuffle(list_of_training_files)
#########################################
#BDT parameters

noise_dim = 100
evaluate_interval = 250 
evaluation_batch_size = 100000
start_time = time.time()
total_time = 0
completed_intervals = 0


for epoch in range(int(1E30)):

	for file in list_of_training_files:

		print('Loading initial training file:',file,'...')

		X_train = np.load(file)
		print(np.shape(X_train))

		X_train = np.take(X_train,np.random.permutation(X_train.shape[0]),axis=0,out=X_train)

		# insert some preprocessing here - needs to work the same over all training files
		# will need to use the same steps in reverse to convert GAN output back to real values
		for index in range(2,8):
			X_train[:,index] = (X_train[:,index] - np.amin(X_train[:,index]))
			X_train[:,index] = (X_train[:,index]/np.amax(X_train[:,index]))
			X_train[:,index] = (X_train[:,index] * 2. ) - 1.
		X_train = X_train[:,2:]

		print('Train images shape -',np.shape(X_train))

		list_for_np_choice = np.arange(np.shape(X_train)[0])

		X_train = np.expand_dims(X_train,1).astype("float32")

		train_dataset = (
			tf.data.Dataset.from_tensor_slices(X_train).batch(batch_size,drop_remainder=True).repeat(1)
		)

		
		for images_for_batch in train_dataset:
        
			if iteration % 250 == 0: 
				print('Iteration:',iteration)
				noise_size = 1000
				gen_noise = np.random.normal(0, 1, (noise_size, 100))
				generated_images = generator.predict(np.expand_dims(gen_noise, 1))
				generated_images = np.squeeze(generated_images)
				# Select a random batch of real images
				idx = np.random.choice(np.arange(X_train.shape[0]), size=noise_size, replace=False)
				real_images_batch = X_train[idx]
				if real_images_batch.ndim > 2:
					real_images_batch = real_images_batch.reshape(real_images_batch.shape[0], -1)
				if generated_images.ndim > 2:
					generated_images = generated_images.reshape(generated_images.shape[0], -1)
	

				# Calculate FID score using the defined function and the batches of real and fake images
				fid_score = calculate_fid(real_images_batch, generated_images)
				# Store the FID score with the corresponding iteration number
				fid_scores.append((iteration, fid_score))
				print(f'FID score at iteration {iteration}: {fid_score}')
		    
			iteration += 1

			gen_loss, disc_loss = train_step(images_for_batch)

			loss_list = np.append(loss_list, [[iteration, gen_loss.numpy(), disc_loss.numpy()]], axis=0)
			
			if (iteration + 1) % evaluate_interval == 0:
				print("Starting XGBoost evaluation.")
				evaluation_noise = np.random.normal(0, 1, (evaluation_batch_size, 1, noise_dim))
				generated_images = generator.predict(evaluation_noise).reshape(evaluation_batch_size, -1)
				# Load real images for evaluation
				real_images_for_evaluation = load_and_preprocess_for_evaluation(evaluation_batch_size)
				# Prepare dataset for XGBoost
				x = np.concatenate((real_images_for_evaluation, generated_images))
				y = np.concatenate((np.ones(evaluation_batch_size), np.zeros(evaluation_batch_size)))
				x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=12345)
				xgb_model = XGBClassifier(
                max_depth=3,  # Maximum tree depth for base learners
                learning_rate=0.1,  # Boosting learning rate (also known as "eta")
                n_estimators=100,  # Number of trees. Equivalent to number of boosting rounds.
				subsample=0.8,  # Subsample ratio of the training instances.
                colsample_bytree=0.8,  # Subsample ratio of columns when constructing each tree.
                gamma=0,  # Minimum loss reduction required to make a further partition on a leaf node.
                min_child_weight=1,  # Minimum sum of instance weight (hessian) needed in a child.
				use_label_encoder=False,  # Avoid using automatic label encoder
                eval_metric='logloss')  # Evaluation metrics for validation data.
				xgb_model.fit(x_train, y_train)

				y_pred = xgb_model.predict(x_test)
				accuracy = accuracy_score(y_test, y_pred)
				print(f"Epoch {iteration + 1}: XGBoost Accuracy = {accuracy:.2f}")

				y_pred_proba = xgb_model.predict_proba(x_test)[:, 1]  # Probability scores for the positive class
				mask = (y_test == 1)
				y_pred_proba_sig = y_pred_proba[mask]
				mask = (y_test == 0)
				y_pred_proba_bkg = y_pred_proba[mask]
				fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
				roc_auc = auc(fpr, tpr)
				# Plot real and generated distributions
				plt.figure(figsize=(8, 6))
				# Plot real data distribution
				plt.hist(y_pred_proba_sig, bins=50, alpha=0.5, color='blue', label='Real Data')
				# Plot generated data distribution
				plt.hist(y_pred_proba_bkg, bins=50, alpha=0.5, color='red', label='Generated Data')
				plt.xlabel('Feature Value')
				plt.ylabel('Frequency')
				plt.title('Real vs. Generated Data Distributions')
				plt.legend()
				# Save the plot into the correct directory
				plt.savefig('%s/Distributions_comparison_iteration_%d.png' % (saving_directory, iteration), bbox_inches='tight')
				plt.close('all')
				# Plot ROC curve
				plt.figure()
				plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
				plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
				plt.xlim([0.0, 1.0])
				plt.ylim([0.0, 1.05])
				plt.xlabel('False Positive Rate')
				plt.ylabel('True Positive Rate')
				plt.title('Receiver Operating Characteristic')
				plt.legend(loc="lower right")
				#Save ROC curve plot into directory
				plt.savefig('%s/CORRELATIONS_true_current.png' % saving_directory, bbox_inches='tight')
				plt.savefig(f'{saving_directory}/CORRELATIONS_true_iteration_{iteration}.png', bbox_inches='tight')
				plt.close('all')
				


                              
				

			# if iteration % save_interval == 0 and iteration > 0:
			if iteration % save_interval == 0:

				t1 = time.time()

				total = t1-t0

				training_time += total

				print('Saving at iteration %d...'%iteration)

				plt.figure(figsize=(8, 4))
				plt.subplot(1,2,1)
				plt.plot(loss_list[:,0], loss_list[:,1])
				plt.ylabel('Gen Loss')
				plt.subplot(1,2,2)
				plt.plot(loss_list[:,0], loss_list[:,2])
				plt.ylabel('Disc Loss')
				plt.subplots_adjust(wspace=0.3, hspace=0.3)
				plt.savefig('%s/LOSSES.png'%(saving_directory),bbox_inches='tight')
				plt.close('all')

				noise_size = 1000
									
				gen_noise = np.random.normal(0, 1, (int(noise_size), 100))
				images = generator.predict([np.expand_dims(gen_noise,1)])

				images = np.squeeze(images)
				select_indexes = np.random.choice(np.arange(np.shape(X_train)[0]), size=noise_size)
				samples = X_train[select_indexes,0].copy()
				print(np.shape(images), np.shape(samples))

				plt.figure(figsize=(5*4, 3*4))
				subplot=0
				
				for i in range(0, 6):
					for j in range(i+1, 6):
						subplot += 1
						plt.subplot(3,5,subplot)
						if subplot == 3: plt.title(iteration)
						plt.hist2d(samples[:noise_size,i], samples[:noise_size,j], bins=50,range=[[-1,1],[-1,1]], norm=LogNorm(), cmap=cmp_root)
						plt.xlabel(axis_titles[i])
						plt.ylabel(axis_titles[j])
				plt.subplots_adjust(wspace=0.3, hspace=0.3)
				plt.savefig('%s/CORRELATIONS_true_current.png'%(saving_directory),bbox_inches='tight')
				plt.savefig(f'{saving_directory}/CORRELATIONS_true_iteration_{iteration}.png',bbox_inches='tight')
				plt.close('all')
				
				plt.figure(figsize=(5*4, 3*4))
				subplot=0
				for i in range(0, 6):
					for j in range(i+1, 6):
						subplot += 1
						plt.subplot(3,5,subplot)
						if subplot == 3: plt.title(iteration)
						plt.hist2d(images[:noise_size,i], images[:noise_size,j], bins=50,range=[[-1,1],[-1,1]], norm=LogNorm(), cmap=cmp_root)
						plt.xlabel(axis_titles[i])
						plt.ylabel(axis_titles[j])
				plt.subplots_adjust(wspace=0.3, hspace=0.3)
				plt.savefig('%s/CORRELATIONS_current.png'%(saving_directory),bbox_inches='tight')
				plt.savefig(f'{saving_directory}/CORRELATIONS_iteration_{iteration}.png',bbox_inches='tight')
				plt.close('all')

				plt.figure(figsize=(3*4, 2*4))
				subplot=0
				for i in range(0, 6):
					subplot += 1
					plt.subplot(2,3,subplot)
					if subplot == 2: plt.title(iteration)
					plt.hist([samples[:noise_size,i], images[:noise_size,i]], bins=50,range=[-1,1], label=['Train','GEN'],histtype='step')
					plt.yscale('log')
					plt.xlabel(axis_titles[i])
					if axis_titles[i] == 'StartZ': plt.legend()
				plt.subplots_adjust(wspace=0.3, hspace=0.3)
				plt.savefig('%s/VALUES.png'%(saving_directory),bbox_inches='tight')
				plt.savefig(f'{saving_directory}/VALUES_iteration_{iteration}.png',bbox_inches='tight')
				plt.close('all')

				# generator.save('%s/generator.h5'%(saving_directory))
				generator.save_weights('%s/generator_weights.h5'%(saving_directory))
				# discriminator.save('%s/discriminator.h5'%(saving_directory))
				discriminator.save_weights('%s/discriminator_weights.h5'%(saving_directory))
				end_time = time.time()
				duration = end_time - start_time
				total_time += duration  # Add this interval's duration to the total time
				completed_intervals += 1  # Increment the count of completed intervals
				 # Calculate and report the average time per interval so far
				average_time_per_interval = total_time / completed_intervals
				print(f"Average time per 250 iterations so far: {average_time_per_interval:.2f} seconds")
				print('Training time: %.2f' % duration)

				print('Saving complete.')
				start_time = time.time() # Reset the start time for the next  250 epochs

				t0 = time.time()
	
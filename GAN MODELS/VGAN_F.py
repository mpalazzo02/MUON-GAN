import numpy as np
 
from tensorflow.keras.layers import Input, Flatten, Dense, Reshape, Dropout, Concatenate, Lambda, ReLU, Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from sklearn.metrics import roc_curve, auc
import tensorflow as tf
import time 
_EPSILON = K.epsilon()
 
import matplotlib as mpl
mpl.use('TkAgg')
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import scipy
 
import math
import glob
import time
import shutil
import os
import random
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from pickle import load
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import accuracy_score
import random
import glob
from scipy.ndimage import gaussian_filter1d
 
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
 
def _loss_generator(y_true, y_pred):
    y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
    out = -(K.log(y_pred))
    return K.mean(out, axis=-1)
 
def split_tensor(index, x):
    return Lambda(lambda x : x[:,:,index])(x)
 
print(tf.__version__)
 
# processID = 5
# os.environ["CUDA_VISIBLE_DEVICES"] = "%d"%processID
noise_dim = 100 
batch_size = 100
save_interval = 250
saving_directory = 'output/'
 
G_architecture = [250,250,50]
D_architecture = [250,250,50]
 
list_of_training_files = glob.glob('SPLIT_*.npy')
 
print(list_of_training_files)
 

print(' ')
print('Initializing networks...')
print(' ')
 

gen_optimizer = tf.keras.optimizers.Adam(lr=0.0004, beta_1=0.5, amsgrad=True)
disc_optimizer = tf.keras.optimizers.Adam(lr=0.0004, beta_1=0.5, amsgrad=True)
 
kernel_initializer_choice='random_uniform'
bias_initializer_choice='random_uniform'
 
##############################################################################################################
# Build Generative model ...
input_noise = Input(shape=(1,100))
 
H = Dense(int(G_architecture[0]))(input_noise)
H = LeakyReLU(alpha=0.2)(H)
H = BatchNormalization(momentum=0.8)(H)
 
for layer in G_architecture[1:]:
 
    H = Dense(int(layer))(H)
    H = LeakyReLU(alpha=0.2)(H)
    H = BatchNormalization(momentum=0.8)(H)
 
H = Dense(6,activation='tanh')(H)
 
generator = Model(inputs=[input_noise], outputs=[H])
generator.summary()
##############################################################################################################


        
##############################################################################################################
# Build Discriminator model ...
d_input = Input(shape=(1,6))
 
H = Flatten()(d_input)
 
for layer in D_architecture:
    H = Dense(int(layer))(H)
    H = LeakyReLU(alpha=0.2)(H)
    H = Dropout(0.2)(H)
 
d_output = Dense(1, activation='sigmoid')(H)
 
discriminator = Model(d_input, [d_output])
discriminator.summary()
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

 
@tf.function
def train_step(images):
 
    noise = tf.random.normal([batch_size, 1, 100])
 
    generated_images = generator([noise])
 
    in_values = tf.concat([generated_images, images],0)
    labels_D_0 = tf.zeros((batch_size, 1))
    labels_D_1 = tf.ones((batch_size, 1))
    labels_D = tf.concat([labels_D_0, labels_D_1],0)
 
    with tf.GradientTape(persistent=True) as disc_tape:
        out_values_choice = discriminator(in_values, training=True)
        disc_loss = tf.keras.losses.binary_crossentropy(tf.squeeze(labels_D),tf.squeeze(out_values_choice))
 
    noise_stacked = tf.random.normal((batch_size, 1, 100), 0, 1)
    labels_stacked = tf.ones((batch_size, 1))
 
    with tf.GradientTape(persistent=True) as gen_tape:
        fake_images2 = generator([noise_stacked], training=True)
        stacked_output_choice = discriminator(fake_images2)
        gen_loss = _loss_generator(tf.squeeze(labels_stacked),tf.squeeze(stacked_output_choice))
 
    grad_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gen_optimizer.apply_gradients(zip(grad_gen, generator.trainable_variables))
 
    grad_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    disc_optimizer.apply_gradients(zip(grad_disc, discriminator.trainable_variables))
 
    return gen_loss, disc_loss
 

start = time.time()
 
iteration = -1
 
loss_list = np.empty((0,3))
 
axis_titles = ['StartX', 'StartY', 'StartZ', 'Px', 'Py', 'Pz']
 
training_time = 0
 
t0 = time.time()
 
random.shuffle(list_of_training_files)
evaluate_interval = 250 
evaluation_batch_size = 100000
start_time = time.time()
total_time = 0
completed_intervals = 0
for epoch in range(int(1E30)):
    noise = np.random.normal(0, 1, (batch_size, 1, noise_dim))
    
    
    
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
 
            if iteration % 250 == 0: print('Iteration:',iteration)
           
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
                X = np.concatenate((real_images_for_evaluation, generated_images))
                       
                y = np.concatenate((np.ones(evaluation_batch_size), np.zeros(evaluation_batch_size)))
            
                # Train and evaluate XGBoost classifier
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=12345)
                
                
                
                xgb_model = XGBClassifier(
                max_depth=3,  # Maximum tree depth for base learners.
                learning_rate=0.1,  # Boosting learning rate (also known as "eta")
                n_estimators=100,  # Number of trees. Equivalent to number of boosting rounds.
                subsample=0.8,  # Subsample ratio of the training instances.
                colsample_bytree=0.8,  # Subsample ratio of columns when constructing each tree.
                gamma=0,  # Minimum loss reduction required to make a further partition on a leaf node.
                min_child_weight=1,  # Minimum sum of instance weight (hessian) needed in a child.
                use_label_encoder=False,  # Avoid using automatic label encoder
                eval_metric='logloss')  # Evaluation metrics for validation data.
                xgb_model.fit(X_train, y_train)
                
                y_pred = xgb_model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                print(f"Epoch {iteration + 1}: XGBoost Accuracy = {accuracy:.2f}")
              

             

                y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]  # Probability scores for the positive class

                #y_pred_y_test = np.column_stack((y_pred_proba,y_test))
                mask = (y_test == 1)
                y_pred_proba_sig = y_pred_proba[mask]
                mask = (y_test == 0)
                y_pred_proba_bkg = y_pred_proba[mask]
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                y_pred_proba = xgb_model.predict_proba(X_test)  # Probability scores for the positive class

                # Separate probabilities for real and generated (fake) classes
                prob_real = y_pred_proba[:, 1]  #  column 1 corresponds to the real class
                prob_generated = y_pred_proba[:, 0]  #  column 0 corresponds to the generated class
                

                # Analyze where the model is extremely confident
                high_confidence_real_indices = np.where(prob_real > 0.9)[0]  # High confidence in real
                high_confidence_generated_indices = np.where(prob_generated > 0.9)[0]  # High confidence in generated

                print(f"High confidence in real data: {len(high_confidence_real_indices)} samples")
                print(f"High confidence in generated data: {len(high_confidence_generated_indices)} samples")
                # Extract high-confidence samples for real and generated data
                high_confidence_real_features = X_test[high_confidence_real_indices]
                high_confidence_generated_features = X_test[high_confidence_generated_indices]

                # Number of features to examine
                num_features = high_confidence_real_features.shape[1]

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
                
                # Plot full distributions for each feature
                for i in range(num_features):
                    plt.figure(figsize=(10, 6))

    # Plot full distribution of feature values for real samples
                    plt.hist(X_test[:, i][y_test == 1], bins=30, alpha=0.5, label='Real', density=True, color='blue')

    # Plot full distribution of feature values for generated samples
                    plt.hist(X_test[:, i][y_test == 0], bins=30, alpha=0.5, label='Generated', density=True, color='red')

                    plt.title(f'Full Feature {i} Distribution')
                    plt.xlabel(f'Feature {i} Value')
                    plt.ylabel('Density')
                    plt.legend()
                    plt.show()
                    # Save full distribution plot into directory
                    full_dist_plot_filename = f'{saving_directory}/Full_Feature_{i}_Distribution_Iteration_{iteration}.png'
                    plt.savefig(full_dist_plot_filename, bbox_inches='tight')
                    plt.close('all')

# Plot high-confidence distributions for each feature
                for i in range(num_features):
                    plt.figure(figsize=(10, 6))

    # Plot distribution of feature values for high-confidence real samples
                    plt.hist(high_confidence_real_features[:, i], bins=30, alpha=0.5, label='High Confidence Real', density=True, color='green')

    # Plot distribution of feature values for high-confidence generated samples
                    plt.hist(high_confidence_generated_features[:, i], bins=30, alpha=0.5, label='High Confidence Generated', density=True, color='orange')

                    plt.title(f'High Confidence Feature {i} Distribution')
                    plt.xlabel(f'Feature {i} Value')
                    plt.ylabel('Density')
                    plt.legend()
                    plt.show()
                    # Save high confidence distribution plot into directory
                    high_conf_dist_plot_filename = f'{saving_directory}/High_Confidence_Feature_{i}_Distribution_Iteration_{iteration}.png'
                    plt.savefig(high_conf_dist_plot_filename, bbox_inches='tight')
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
                images = generator.predict([np.expand_dims(gen_noise, 1)])
                images = np.squeeze(images)
                # Check the shape of samples after squeezing images
                


               
                select_indexes = np.random.choice(np.arange(np.shape(X_train)[0]), size=noise_size)
                samples = X_train[select_indexes].copy()
                samples = np.squeeze(samples)
                print("Shape of samples:", samples.shape)
                
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
 

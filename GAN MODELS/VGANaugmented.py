import numpy as np
from scipy.ndimage import gaussian_filter1d
from tensorflow.keras.layers import (
    Input, Flatten, Dense, Reshape, Dropout, Concatenate, Lambda, ReLU, Activation, BatchNormalization, LeakyReLU
)
from tensorflow.keras.models import Model
from sklearn.metrics import roc_curve, auc
import tensorflow as tf
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import random
import glob
import os

# Initialisation
mpl.use('TkAgg')
mpl.use('Agg')
_EPSILON = tf.keras.backend.epsilon()
noise_dim = 100
batch_size = 100
save_interval = 250
saving_directory = 'output/'
G_architecture = [250, 250, 50]
D_architecture = [250, 250, 50]
list_of_training_files = glob.glob('SPLIT_*.npy')

# Generator loss function
def _loss_generator(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, _EPSILON, 1.0 - _EPSILON)
    out = -(tf.math.log(y_pred))
    return tf.reduce_mean(out, axis=-1)

# Split tensor function
def split_tensor(index, x):
    return Lambda(lambda x: x[:, :, index])(x)

# Preprocess data function
def preprocess_data(X):
    for index in range(4, 8):
        X[:, index] = (X[:, index] - np.amin(X[:, index]))
        X[:, index] = (X[:, index] / np.amax(X[:, index]))
        X[:, index] = (X[:, index] * 2.) - 1.
    for index in range(2, 4):
        X[:, index] = gaussian_filter1d(X[:, index], sigma=4)
    return X[:, 2:]

# Load and preprocess data for evaluation
def load_and_preprocess_for_evaluation(batch_size):
    file_index = np.random.randint(0, len(list_of_training_files))
    X = np.load(list_of_training_files[file_index])
    X = preprocess_data(X)
    indices = np.random.permutation(len(X))[:batch_size]
    return X[indices]

# Get XGBoost importance scores
def get_xgb_importance_scores(generator, batch_size, noise_dim):
    evaluation_noise = np.random.normal(0, 1, (batch_size, 1, noise_dim))
    generated_images = generator.predict(evaluation_noise).reshape(batch_size, -1)
    real_images_for_evaluation = load_and_preprocess_for_evaluation(batch_size)
    X = np.concatenate((real_images_for_evaluation, generated_images))
    y = np.concatenate((np.ones(batch_size), np.zeros(batch_size)))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=12345)
    xgb_model = XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100, subsample=0.8,
                              colsample_bytree=0.8, gamma=0, min_child_weight=1, use_label_encoder=False,
                              eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    return xgb_model.feature_importances_

# Train step function
def train_step(images, importance_scores=None):
    noise = tf.random.normal([batch_size, 1, noise_dim])
    if importance_scores is not None:
        importance_weights = tf.constant(importance_scores, dtype=tf.float32)
        emphasis_factor = tf.constant(1.5)
        adjusted_noise = noise * (1 + tf.reshape(importance_weights, (1, -1)) * emphasis_factor)
    else:
        adjusted_noise = noise
    generated_images = generator([adjusted_noise], training=True)
    in_values = tf.concat([generated_images, images], 0)
    labels_D_0 = tf.zeros((batch_size, 1))
    labels_D_1 = tf.ones((batch_size, 1))
    labels_D = tf.concat([labels_D_0, labels_D_1], 0)
    with tf.GradientTape(persistent=True) as disc_tape:
        out_values_choice = discriminator(in_values, training=True)
        disc_loss = tf.keras.losses.binary_crossentropy(tf.squeeze(labels_D), tf.squeeze(out_values_choice))
    noise_stacked = tf.random.normal([batch_size, 1, 100])
    with tf.GradientTape() as gen_tape:
        fake_images2 = generator([noise_stacked], training=True)
        stacked_output_choice = discriminator(fake_images2)
        gen_loss = _loss_generator(tf.squeeze(labels_D_1), tf.squeeze(stacked_output_choice))
    grad_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gen_optimizer.apply_gradients(zip(grad_gen, generator.trainable_variables))
    grad_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    disc_optimizer.apply_gradients(zip(grad_disc, discriminator.trainable_variables))
    return gen_loss, disc_loss

# Main training loop
evaluate_interval_b = 50
start = time.time()
iteration = -1
loss_list = np.empty((0, 3))
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
        print('Loading initial training file:', file, '...')
        X_train = np.load(file)
        print(np.shape(X_train))
        X_train = np.take(X_train, np.random.permutation(X_train.shape[0]), axis=0, out=X_train)
        for index in range(4, 8):
            X_train[:, index] = (X_train[:, index] - np.amin(X_train[:, index]))
            X_train[:, index] = (X_train[:, index] / np.amax(X_train[:, index]))
            X_train[:, index] = (X_train[:, index] * 2.) - 1.
        for index in range(2, 4):
            X_train[:, index] = gaussian_filter1d(X_train[:, index], sigma=4)
        X_train = X_train[:, 2:]
        print('Train images shape -', np.shape(X_train))
        list_for_np_choice = np.arange(np.shape(X_train)[0])
        X_train = np.expand_dims(X_train, 1).astype("float32")
        train_dataset = (
            tf.data.Dataset.from_tensor_slices(X_train).batch(batch_size, drop_remainder=True).repeat(1)
        )
        for images_for_batch in train_dataset:
            if iteration % evaluate_interval_b == 0:
                importance_scores = get_xgb_importance_scores(generator, evaluation_batch_size, noise_dim)
            if iteration % 250 == 0:
                print('Iteration:', iteration)
            iteration += 1
            gen_loss, disc_loss = train_step(images_for_batch)
            loss_list = np.append(loss_list, [[iteration, gen_loss.numpy(), disc_loss.numpy()]], axis=0)
            if (iteration + 1) % evaluate_interval == 0:
                print("Starting XGBoost evaluation.")
                evaluation_noise = np.random.normal(0, 1, (evaluation_batch_size, 1, noise_dim))
                generated_images = generator.predict(evaluation_noise).reshape(evaluation_batch_size, -1)
                real_images_for_evaluation = load_and_preprocess_for_evaluation(evaluation_batch_size)
                X = np.concatenate((real_images_for_evaluation, generated_images))
                y = np.concatenate((np.ones(evaluation_batch_size), np.zeros(evaluation_batch_size)))
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=12345)
                xgb_model = XGBClassifier(
                    max_depth=3, learning_rate=0.1, n_estimators=100, subsample=0.8, colsample_bytree=0.8, gamma=0,
                    min_child_weight=1, use_label_encoder=False, eval_metric='logloss'
                )
                xgb_model.fit(X_train, y_train)
                y_pred = xgb_model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                print(f"Epoch {iteration + 1}: XGBoost Accuracy = {accuracy:.2f}")
                y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
                mask = (y_test == 1)
                y_pred_proba_sig = y_pred_proba[mask]
                mask = (y_test == 0)
                y_pred_proba_bkg = y_pred_proba[mask]
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                y_pred_proba = xgb_model.predict_proba(X_test)
                prob_real = y_pred_proba[:, 1]
                prob_generated = y_pred_proba[:, 0]
                importance_scores = xgb_model.feature_importances_
                print("Feature Importances:")
                for i, score in enumerate(importance_scores):
                    print(f'Feature {i}: Importance Score: {score}')
                high_confidence_real_indices = np.where(prob_real > 0.9)[0]
                high_confidence_generated_indices = np.where(prob_generated > 0.9)[0]
                print(f"High confidence in real data: {len(high_confidence_real_indices)} samples")
                print(f"High confidence in generated data: {len(high_confidence_generated_indices)} samples")
            if iteration % save_interval == 0:
                t1 = time.time()
                total = t1 - t0
                training_time += total
                print('Saving at iteration %d...' % iteration)
                plt.figure(figsize=(8, 4))
                plt.subplot(1, 2, 1)
                plt.plot(loss_list[:, 0], loss_list[:, 1])
                plt.ylabel('Gen Loss')
                plt.subplot(1, 2, 2)
                plt.plot(loss_list[:, 0], loss_list[:, 2])
                plt.ylabel('Disc Loss')
                plt.subplots_adjust(wspace=0.3, hspace=0.3)
                plt.savefig('%s/LOSSES.png' % (saving_directory), bbox_inches='tight')
                plt.close('all')
                noise_size = 1000
                gen_noise = np.random.normal(0, 1, (int(noise_size), 100))
                images = generator.predict([np.expand_dims(gen_noise, 1)])
                images = np.squeeze(images)
                select_indexes = np.random.choice(np.arange(np.shape(X_train)[0]), size=noise_size)
                samples = X_train[select_indexes].copy()
                samples = np.squeeze(samples)
                print("Shape of samples:", samples.shape)
                print(np.shape(images), np.shape(samples))
                plt.figure(figsize=(5 * 4, 3 * 4))
                subplot = 0
                plt.figure(figsize=(3 * 4, 2 * 4))
                subplot = 0
                for i in range(0, 6):
                    subplot += 1
                    plt.subplot(2, 3, subplot)
                    if subplot == 2:
                        plt.title(iteration)
                    plt.hist([samples[:noise_size, i], images[:noise_size, i]], bins=50, range=[-1, 1],
                             label=['Train', 'GEN'], histtype='step')
                    plt.yscale('log')
                    plt.xlabel(axis_titles[i])
                    if axis_titles[i] == 'StartZ':
                        plt.legend()
                plt.subplots_adjust(wspace=0.3, hspace=0.3)
                plt.savefig('%s/VALUES.png' % (saving_directory), bbox_inches='tight')
                plt.savefig(f'{saving_directory}/VALUES_iteration_{iteration}.png', bbox_inches='tight')
                plt.close('all')
                generator.save_weights('%s/generator_weights.h5' % (saving_directory))
                discriminator.save_weights('%s/discriminator_weights.h5' % (saving_directory))
                end_time = time.time()
                duration = end_time - start_time
                total_time += duration
                completed_intervals += 1
                average_time_per_interval = total_time / completed_intervals
                print(f"Average time per 250 iterations so far: {average_time_per_interval:.2f} seconds")
                print('Training time: %.2f' % duration)
                print('Saving complete.')
                start_time = time.time()
                t0 = time.time()

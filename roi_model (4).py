import tensorflow as tf
from keras import optimizers
from keras.callbacks import TensorBoard, EarlyStopping
from keras.models import Sequential, Model
from keras.layers import Dense, Input, concatenate, Multiply, Activation
from keras.optimizers import Adam, SGD
#from keras.layers.core import Lambda, Dropout
from keras.layers import Lambda, Dropout

from keras import backend as K
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras import regularizers
import keras

#from keras.engine.topology import Layer
from keras.layers import Layer


from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints

from tensorflow.keras.layers import Dropout
from tensorflow.keras.losses import binary_crossentropy


def get_uplift_rank_criteo_model():
    feature_input = Input(shape=(12,), name="p0_raw_features")
    treated_input = Input(shape=(1,), name="treated_input")
    reward_input = Input(shape=(1,), name="reward_input") 
    
    p1_hidden_1 = Dense(8, activation="relu", name="p1_hidden_1", kernel_regularizer=regularizers.l2(1e-5))(feature_input)
    
    p1_output =  Dense(1, name="p1", kernel_regularizer=regularizers.l2(1e-5))(p1_hidden_1)


    final_model = Model(inputs=[feature_input,treated_input, reward_input], outputs=p1_output)
    
    p_output = tf.exp(p1_output) / tf.reduce_sum(tf.exp(p1_output))
    
    q_output = tf.math.log(p_output)
    
    r_output = tf.reduce_sum(reward_input * q_output * treated_input) / tf.reduce_sum(treated_input) - tf.reduce_sum(reward_input * q_output * (1 - treated_input)) / tf.reduce_sum(1 - treated_input)
    
    loss = 0.0 - r_output
    
    final_model.add_loss(loss)
    final_model.add_metric(loss, name='obj')
    return final_model


def get_roi_rank_criteo_model_with_counterfactual():
    
    feature_input = Input(shape=(12,), name="p0_raw_features")
    treated_input = Input(shape=(1,), name="treated_input")
    reward_input = Input(shape=(1,), name="reward_input")
    cost_input = Input(shape=(1,), name="cost_input")

    reg = 2.5e-5
    # shared layer
    p1_share_1 = Dense(8, activation="relu", name="share", kernel_regularizer=regularizers.l2(reg))(feature_input)
    
    
    
    # propensity score loss
    p2_hidden_1 =  Dense(8, activation="relu", name="p2_hidden_1", kernel_regularizer=regularizers.l2(reg))(p1_share_1)
    ps_output =  Dense(1, activation="sigmoid", name="p2", kernel_regularizer=regularizers.l2(reg))(p2_hidden_1)
    loss_ps = tf.reduce_mean(binary_crossentropy(treated_input, ps_output))
    
    # loss_100
    p3_hidden_1 =  Dense(8, activation="relu", name="p3_hidden_1", kernel_regularizer=regularizers.l2(reg))(p1_share_1)
    f0000_output =  Dense(1, activation="sigmoid", name="p3", kernel_regularizer=regularizers.l2(reg))(p3_hidden_1)
    label_100 = tf.cast(tf.logical_and(tf.equal(treated_input, 1), tf.logical_and(tf.equal(cost_input, 0), tf.equal(reward_input, 0))), dtype=tf.float32)
    loss_100 = tf.reduce_mean(binary_crossentropy(label_100, f0000_output*ps_output))
    
    # loss_110
    p5_hidden_1 =  Dense(8, activation="relu", name="p5_hidden_1", kernel_regularizer=regularizers.l2(reg))(p1_share_1)
    f0100_output =  Dense(1, activation="sigmoid", name="p5", kernel_regularizer=regularizers.l2(reg))(p5_hidden_1)
    label_110 = tf.cast(tf.logical_and(tf.equal(treated_input, 1), tf.logical_and(tf.equal(cost_input, 1), tf.equal(reward_input, 0))), dtype=tf.float32)
    loss_110 = tf.reduce_mean(binary_crossentropy(label_110, f0100_output*ps_output))

    
    # loss_111
    p6_hidden_1 =  Dense(8, activation="relu", name="p6_hidden_1", kernel_regularizer=regularizers.l2(reg))(p1_share_1)
    f0101_output =  Dense(1, activation="sigmoid", name="p6", kernel_regularizer=regularizers.l2(reg))(p6_hidden_1)
    label_111 = tf.cast(tf.logical_and(tf.equal(treated_input, 1), tf.logical_and(tf.equal(cost_input, 1), tf.equal(reward_input, 1))), dtype=tf.float32)
    loss_111 = tf.reduce_mean(binary_crossentropy(label_111, f0101_output*ps_output))
    
    # loss_000
    label_000 = tf.cast(tf.logical_and(tf.equal(treated_input, 0), tf.logical_and(tf.equal(cost_input, 0), tf.equal(reward_input, 0))), dtype=tf.float32)
    loss_000 = tf.reduce_mean(binary_crossentropy(label_000, (f0000_output + f0100_output + f0101_output)*(1-ps_output)))
    
    # reward-unware loss
    p1_hidden_1 = Dense(8, activation="relu", name="p1_hidden_1", kernel_regularizer=regularizers.l2(1e-5))(p1_share_1)
    p1_output =  Dense(1, name="p1", kernel_regularizer=regularizers.l2(1e-5))(p1_hidden_1)
    reward_unware_output = tf.exp(p1_output) / tf.reduce_sum(tf.exp(p1_output))
    q_output = tf.math.log(reward_unware_output)
    r_output = tf.reduce_sum(cost_input * q_output * treated_input) / tf.reduce_sum(treated_input) - tf.reduce_sum(cost_input * q_output * (1 - treated_input)) / tf.reduce_sum(1 - treated_input)
    loss_reward_unware = 0.0 - r_output
    
    # roi loss
    p4_hidden_1 =  Dense(8, activation="relu", name="p4_hidden_1", kernel_regularizer=regularizers.l2(reg))(p1_share_1)
    roi_output =  Dense(1, activation="sigmoid", name="p4", kernel_regularizer=regularizers.l2(reg))(p4_hidden_1)
    qr = tf.math.log(roi_output / (1 - roi_output))
    qc = tf.math.log(1 - roi_output)
    r_output = tf.reduce_sum(reward_input * qr * treated_input) / tf.reduce_sum(treated_input) - tf.reduce_sum(reward_input * qr * (1 - treated_input)) / tf.reduce_sum(1 - treated_input)
    c_output = tf.reduce_sum(cost_input * qc * treated_input) / tf.reduce_sum(treated_input) - tf.reduce_sum(cost_input * qc * (1 - treated_input)) / tf.reduce_sum(1 - treated_input)
    loss_roi = - (r_output + c_output)
    
    
    final_model = Model(inputs=[feature_input, treated_input, reward_input, cost_input], outputs=[reward_unware_output, roi_output, ps_output, f0000_output, f0100_output, f0101_output])
    loss_reward_unware=40*loss_reward_unware
    loss_roi=100*loss_roi
    loss_ps=10*loss_ps
    loss_100=10*loss_100
    loss_110=20*loss_110
    loss_111=14*loss_111
    loss_000=10*loss_000
    loss = loss_reward_unware + loss_roi + loss_ps + loss_100 + loss_110 + loss_111 + loss_000
    #loss = loss_roi
    final_model.add_loss(loss)
    final_model.add_metric(loss_roi, name='loss_roi')
    final_model.add_metric(loss_reward_unware, name='loss_reward_unware')
    final_model.add_metric(loss_ps, name='loss_ps')
    final_model.add_metric(loss_100, name='loss_100')
    final_model.add_metric(loss_110, name='loss_110')
    final_model.add_metric(loss_111, name='loss_111')
    final_model.add_metric(loss_000, name='loss_000')
    return final_model


def get_roi_rank_criteo_model_with_bcauss():
    
    feature_input = Input(shape=(12,), name="p0_raw_features")
    treated_input = Input(shape=(1,), name="treated_input")
    reward_input = Input(shape=(1,), name="reward_input")
    cost_input = Input(shape=(1,), name="cost_input")

    reg = 2.5e-5
    p1_hidden_1 = Dense(8, activation="relu", name="p1_hidden_1", kernel_regularizer=regularizers.l2(reg))(feature_input)
    
    
    ########  loss
    
    p2_hidden_1 =  Dense(8, activation="relu", name="p2_hidden_1", kernel_regularizer=regularizers.l2(reg))(p1_hidden_1)
    
    p3_hidden_1 =  Dense(8, activation="relu", name="p3_hidden_1", kernel_regularizer=regularizers.l2(reg))(p2_hidden_1)
    t_predictions =  Dense(1, activation="sigmoid", name="t1", kernel_regularizer=regularizers.l2(reg))(p3_hidden_1)
    
    p4_hidden_1 =  Dense(8, activation="relu", name="p4_hidden_1", kernel_regularizer=regularizers.l2(reg))(p3_hidden_1)
    q_output =  Dense(1, activation="sigmoid", name="p1", kernel_regularizer=regularizers.l2(reg))(p4_hidden_1)
    
    t_pred = (t_predictions + 0.001) / 1.002
    
    ##  self-supervised covariate balancing objective
    ones_to_sum = K.repeat_elements(treated_input / t_pred, rep=12, axis=1)*feature_input
    zeros_to_sum = K.repeat_elements((1 - treated_input) / (1 - t_pred), rep=12, axis=1)*feature_input
    ones_mean = tf.math.reduce_sum(ones_to_sum,0)/tf.math.reduce_sum(treated_input / t_pred,0)
    zeros_mean = tf.math.reduce_sum(zeros_to_sum,0)/tf.math.reduce_sum((1 - treated_input) / (1 - t_pred),0)
    
    
    final_model = Model(inputs=[feature_input, treated_input, reward_input, cost_input], outputs=q_output)
    
    
    qr = tf.math.log(q_output / (1 - q_output))
    qc = tf.math.log(1 - q_output)
    
    r_output = tf.reduce_sum(reward_input * qr * treated_input) / tf.reduce_sum(treated_input) - tf.reduce_sum(reward_input * qr * (1 - treated_input)) / tf.reduce_sum(1 - treated_input)
    c_output = tf.reduce_sum(cost_input * qc * treated_input) / tf.reduce_sum(treated_input) - tf.reduce_sum(cost_input * qc * (1 - treated_input)) / tf.reduce_sum(1 - treated_input)

    loss = - (r_output + c_output) + 0.01*tf.keras.losses.mean_squared_error(zeros_mean, ones_mean)
    
    final_model.add_loss(loss)
    final_model.add_metric(loss, name='obj')
    return final_model


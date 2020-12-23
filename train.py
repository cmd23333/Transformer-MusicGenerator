import tensorflow as tf
import numpy as np
import random
from util.data import generate_batch_song
from model.model import Encoder


def music_model(seq_len, unique_notes, d_model=128):
    inputs = tf.keras.layers.Input(shape=(seq_len,))
    forward_pass = Encoder(num_layers=6, d_model=d_model, num_heads=4, dff=4 * d_model,
                           input_vocab_size=unique_notes + 1,
                           maximum_position_encoding=50, rate=0.1)(inputs, mask=None)
    outputs = tf.keras.layers.Dense(unique_notes + 1, activation="softmax")(forward_pass)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='generate_scores_rnn')
    return model


class TrainModel:

    def __init__(self, epochs, note_tokenizer, list_all_midi, frame_per_second,
                 batch_nnet_size, batch_song, optimizer, loss_fn, total_songs, model, seq_len):
        self.epochs = epochs
        self.note_tokenizer = note_tokenizer
        self.list_all_midi = list_all_midi
        self.frame_per_second = frame_per_second
        self.batch_nnet_size = batch_nnet_size
        self.batch_song = batch_song
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.total_songs = total_songs
        self.model = model
        self.seq_len = seq_len

    def train(self):
        loss_list = []
        steps_nnet = 0
        for epoch in range(self.epochs):
            # for each epochs, we shufle the list of all the datasets
            random.shuffle(self.list_all_midi)
            steps = 0
            # We will iterate all songs by self.song_size
            for i in range(0, self.total_songs, self.batch_song):
                steps += 1
                inputs_nnet_large, outputs_nnet_large = generate_batch_song(
                    self.list_all_midi, self.batch_song, start_index=i, fs=self.frame_per_second,
                    seq_len=self.seq_len)  # We use the function that have been defined here
                inputs_nnet_large = np.array(self.note_tokenizer.transform(inputs_nnet_large), dtype=np.int32)
                outputs_nnet_large = np.array(self.note_tokenizer.transform(outputs_nnet_large), dtype=np.int32)
                index_shuffled = np.arange(start=0, stop=len(inputs_nnet_large))
                np.random.shuffle(index_shuffled)

                for nnet_steps in range(0, len(index_shuffled), self.batch_nnet_size):
                    steps_nnet += 1
                    current_index = index_shuffled[nnet_steps:nnet_steps + self.batch_nnet_size]
                    inputs_nnet, outputs_nnet = inputs_nnet_large[current_index], outputs_nnet_large[current_index]

                    # To make sure no exception thrown by tensorflow on autograph
                    if len(inputs_nnet) // self.batch_nnet_size != 1:
                        break
                    loss = self.train_step(inputs_nnet, outputs_nnet)
                    loss_list.append(tf.math.reduce_sum(loss))
                    if steps_nnet % 20 == 0:
                        print(
                            "epochs {} | Steps {} | loss : {}".format(epoch + 1, steps_nnet, tf.math.reduce_sum(loss)))
        return loss_list

    @tf.function
    def train_step(self, inputs, targets):
        with tf.GradientTape() as tape:
            prediction = self.model(inputs)
            prediction = prediction[:, -1, :]
            loss = self.loss_fn(targets, prediction)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss
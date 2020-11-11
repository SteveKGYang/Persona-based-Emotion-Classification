import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from transformers import BertTokenizer, TFBertModel, BertConfig, TFBertForPreTraining
from tools import *
from random import shuffle
import logging
import os

from post_train_data_preprocess import *
import argparse

logging.disable(30)

parser = argparse.ArgumentParser(description='Utility Functions')
parser.add_argument('-emory_file', type=str, default="data/emotion-detection-trn.json")
parser.add_argument('-friends_file', type=str, default="./Friends/friends.augmented.json")
parser.add_argument('-emotion_push_file', type=str, default="emotionpush.augmented.json")
parser.add_argument('-save_path', type=str, default='./post_train_model/')
parser.add_argument('-batch_size', type=int, default=2)
parser.add_argument('-epoch', type=int, default=50)
parser.add_argument('-max_lr', type=float, default=1e-4)
parser.add_argument('-max_grad_norm', type=float, default=0.01, help="prefix of .dict and .labels files")
parser.add_argument("--adam_epsilon", default=1e-7, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--warm_up_steps", default=1500, type=int, help="Epsilon for Adam optimizer.")
parser.add_argument("--learning_rate_step", default=2000, type=int, help="Epsilon for Adam optimizer.")

args = parser.parse_args()


class PostTrainBert(keras.Model):
    def __init__(self):
        super(PostTrainBert, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = TFBertForPreTraining.from_pretrained('bert-base-uncased')

    def get_tokenizer(self):
        return self.tokenizer

    def call(self, inputs, attention_masks=None):
        return self.bert(inputs, attention_mask=attention_masks)


def loss(mlm_pred, nsp_pred, mlm_records, nsp_labels, batch_size):
    with tf.device("/gpu:%d" % 2):
        nsp_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=nsp_labels, logits=nsp_pred))
        mlm_loss = 0.
        num = 0
        for i in range(min(batch_size, mlm_pred.get_shape()[0])):
            single_pred = mlm_pred[i]
            single_records = mlm_records[i]
            for record in single_records:
                pred = single_pred[record[0]]
                loss = -(tf.reduce_sum(
                    tf.math.log(tf.clip_by_value(1 - pred[0:record[1]], 1e-8, 1.0))) + tf.math.log(
                    tf.clip_by_value(pred[record[1]], 1e-8, 1.0)) + tf.reduce_sum(
                    tf.math.log(tf.clip_by_value(1 - pred[record[1] + 1:-1], 1e-8, 1.0))))
                mlm_loss += loss
                num += 1
                # pred_list.append(single_pred[record[0]])
                # y = np.zeros([mlm_pred.get_shape()[2]])
                # y[record[1]] = 1.
                # label_list.append(y)
        if num != 0:
            mlm_loss /= num
            nsp_loss += mlm_loss
        # pred_list = tf.stack(pred_list, axis=0)
        # label_list = tf.stack(label_list, axis=0)
        # mlm_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_list, logits=pred_list))
        return nsp_loss


def exp_decay_with_warmup(warmup_step, learning_rate_base, global_step, learning_rate_step, learning_rate_decay, staircase=False):
    linear_increase = learning_rate_base * tf.cast(global_step / warmup_step, tf.float32)
    exponential_decay = tf.compat.v1.train.exponential_decay(learning_rate_base,
                                                   global_step - warmup_step,
                                                   learning_rate_step,
                                                   learning_rate_decay,
                                                   staircase=staircase)
    learning_rate = tf.cond(global_step <= warmup_step,
                            lambda: linear_increase,
                            lambda: exponential_decay)
    return learning_rate


def train():
    bert_model = PostTrainBert()
    data_maker = PostTrainDataMaker(bert_model.get_tokenizer())
    data = data_maker.create_data(args.emory_file, args.friends_file, args.emotion_push_file)
    global_step = args.epoch * np.ceil(float(len(data)/args.batch_size))
    lr_decayed = exp_decay_with_warmup(args.warm_up_steps, args.max_lr, global_step, args.learning_rate_step, 0.9)
    optimizer = keras.optimizers.Adam(args.max_lr, epsilon=args.adam_epsilon, clipvalue=args.max_grad_norm)
    for p in range(args.epoch):
        total_loss = 0.
        shuffle(data)
        print("Training for epoch {}".format(p))
        for i in range(0, len(data), args.batch_size):
            inputs = []
            attns = []
            next_labels = []
            mlm_records = []
            for j in data[i:min(len(data), i+args.batch_size)]:
                inputs.append(j[0])
                next_labels.append(j[1])
                mlm_records.append(j[2])
                attns.append(j[3])
            inputs = np.array(inputs)
            attns = np.array(attns)
            next_labels = np.array(next_labels)
            with tf.GradientTape() as tape:
                with tf.device("/gpu:%d" % 0):
                    predictions = bert_model(inputs)
                loss_value = loss(tf.nn.softmax(predictions[0]), predictions[1], mlm_records, next_labels, args.batch_size)
            gradients = tape.gradient(loss_value, bert_model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, bert_model.trainable_weights))
            total_loss += loss_value
        print("epoch: {} training loss: {}".format(p, total_loss / len(data)))
        if (p+1) % 5 == 0:
            output_dir = os.path.join(args.save_path, 'checkpoint-{}'.format(p))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            bert_model.bert.save_pretrained(output_dir)
    bert_model.tokenizer.save_pretrained(args.save_path)


if __name__ == "__main__":
    train()


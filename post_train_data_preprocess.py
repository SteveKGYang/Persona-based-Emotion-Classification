#encoding:utf-8
import json
import numpy as np
import random
from transformers import BertTokenizer


class DataLoader:
    def __init__(self):
        pass

    def load_emory_data(self, file_name):
        data_set = []
        with open(file_name, 'r', encoding='utf8')as fp:
            json_data = json.load(fp)
            for episode in json_data["episodes"]:
                for scenes in episode["scenes"]:
                    data = []
                    for uttr in scenes["utterances"]:
                        data.append(uttr["transcript"])
                    data_set.append(data)
        return data_set

    def load_emotionline_data(self, file_name):
        data_set = []
        with open(file_name, 'r', encoding='utf8')as fp:
            json_data = json.load(fp)
            for scene in json_data:
                data = []
                for uttr in scene:
                    data.append(uttr["utterance"])
                data_set.append(data)
        return data_set


class PostTrainDataMaker:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.vocabs = self.tokenizer.get_vocab()
        self.data_loader = DataLoader()
        self.final_data = []

    def create_data(self, emory_file, friends_file, emotionpush_file):
        emory_data = self.data_loader.load_emory_data(emory_file)
        friends_data = self.data_loader.load_emotionline_data(friends_file)
        emotion_push_data = self.data_loader.load_emotionline_data(emotionpush_file)
        dataset = emory_data + friends_data + emotion_push_data
        corpus = []
        max_len = 0
        for data in dataset:
            corpus += data
        for dialogue in dataset:
            for j in range(len(dialogue)-1):
                mlm_labels = []
                isnext = 0.
                sena = dialogue[j]
                if np.random.rand() < 0.5:
                    senb = dialogue[j+1]
                    isnext = 1.
                else:
                    senb = random.choice(corpus)
                if len(sena) == 0 or len(senb) == 0:
                    continue
                sena = sena.lower()
                senb = senb.lower()
                sen = sena + " [SEP] " + senb
                k = self.tokenizer.encode(sen)
                for i, token in enumerate(k):
                    if token not in [101, 102]:
                        if np.random.rand() < 0.15:
                            mlm_labels.append([i, token])
                            c_num = np.random.rand()
                            if c_num <= 0.8:
                                k = k[0:i] + [self.vocabs["[MASK]"]] + k[i + 1:len(k)]
                            elif 0.8 < c_num <= 0.9:
                                rep_word = random.choice(list(self.vocabs.values()))
                                while rep_word in [101, 102, self.vocabs["[MASK]"], token]:
                                    rep_word = random.choice(list(self.vocabs.values()))
                                k = k[0:i] + [rep_word] + k[i + 1:len(k)]
                if len(k) > max_len:
                    max_len = len(k)
                self.final_data.append([k, [isnext, 1-isnext], mlm_labels])
        for data in self.final_data:
            j = len(data[0])
            data[0] += [0]*(max_len-j)
            data.append([1]*j+[0]*(max_len-j))
        return self.final_data


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    a = PostTrainDataMaker(tokenizer)
    a.create_data("./data/emotion-detection-trn.json", "./Friends/friends.augmented.json", "emotionpush.augmented.json")
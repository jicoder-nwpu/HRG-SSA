from transformers import T5Tokenizer
import os
from io_utils import *
import torch
import numpy as np
import random


class Reader():

    def __init__(self, model_path, data_dir, logger, window_size, batch_size, has_dev=True, audio_encoder='hubert_base') -> None:
        self.logger = logger
        self.tokenizer = self.init_tokenizer(model_path)
        self.data_dir = data_dir
        self.window_size = window_size
        self.batch_size = batch_size
        encoded_data_path = os.path.join(self.data_dir, "encoded_data.pkl")
        self.audio_feature_path = os.path.join(self.data_dir, "audio_features_{}.pkl".format(audio_encoder))
        self.video_feature_path = os.path.join(self.data_dir, "video_features.pkl")
        self.audio_encoder = audio_encoder

        if os.path.exists(encoded_data_path):
            self.logger.info("Load encoded data from {}".format(encoded_data_path))
            self.data = load_pickle(encoded_data_path)
        else:
            self.logger.info("Encode data and save to {}".format(encoded_data_path))
            train = self.encode_data("train")
            test = self.encode_data("test")
            if has_dev:
                dev = self.encode_data("dev")
            else:
                dev = self.encode_data("test")

            self.data = {"train": train, "dev": dev, "test": test}
            save_pickle(self.data, encoded_data_path)
        self.train_steps = self.get_num_training_steps_per_epoch('train')

    def get_num_training_steps_per_epoch(self, data_type):
        num = 0
        for n in self.data[data_type]:
            num += int(n) * ((len(self.data[data_type][n]) + self.batch_size - 1) // self.batch_size)
        return num

    def init_tokenizer(self, model_path):
        tokenizer = T5Tokenizer.from_pretrained('./pretrained_model/')

        special_tokens = []

        special_tokens.append('<bos_emotion>')
        special_tokens.append('<eos_emotion>')
        special_tokens.append('<bos_sentiment>')
        special_tokens.append('<eos_sentiment>')
        special_tokens.append('<bos_audio>')
        special_tokens.append('<eos_audio>')
        special_tokens.append('<bos_video>')
        special_tokens.append('<eos_video>')
        special_tokens.append('<eos_status>')
        special_tokens.append('<bos_resp>')

        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

        return tokenizer
    
    @property
    def bos_audio(self):
        return "<bos_audio>"

    @property
    def eos_audio(self):
        return "<eos_audio>"

    @property
    def bos_video(self):
        return "<bos_video>"

    @property
    def eos_video(self):
        return "<eos_video>"

    @property
    def bos_emotion(self):
        return "<bos_emotion>"

    @property
    def eos_emotion(self):
        return "<eos_emotion>"
    
    @property
    def bos_resp(self):
        return "<bos_resp>"

    @property
    def eos_sentiment(self):
        return "<eos_sentiment>"

    @property
    def bos_sentiment(self):
        return "<bos_sentiment>"
    
    @property
    def eos_status(self):
        return "<eos_status>"

    @property
    def pad_token(self):
        return self.tokenizer.pad_token

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    @property
    def eos_token(self):
        return self.tokenizer.eos_token

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def bos_token(self):
        return self.tokenizer.bos_token

    @property
    def bos_token_id(self):
        return self.tokenizer.bos_token_id

    @property
    def unk_token(self):
        return self.tokenizer.unk_token

    @property
    def max_seq_len(self):
        return self.tokenizer.model_max_length

    @property
    def vocab_size(self):
        return len(self.tokenizer)

    def get_token_id(self, token):
        return self.tokenizer.convert_tokens_to_ids(token)
    
    def encode_text(self, text, bos_token=None, eos_token=None):
        tokens = text.split() if isinstance(text, str) else text

        assert isinstance(tokens, list)

        if bos_token is not None:
            if isinstance(bos_token, str):
                bos_token = [bos_token]

            tokens = bos_token + tokens

        if eos_token is not None:
            if isinstance(eos_token, str):
                eos_token = [eos_token]

            tokens = tokens + eos_token

        encoded_text = self.tokenizer.encode(" ".join(tokens))

        return encoded_text

    def collate_fn(self, batch, max_length):
        max_len = 0
        for seq in batch:
            max_len = max(max_len, len(seq))
        # 在这里实现数据的截断和填充
        def pad_sequence(seq, max_len):
            if len(seq) > max_length:
                return torch.tensor([3] + list(seq)[-max_length + 1:])
            return torch.tensor(list(seq) + [self.pad_token_id]*(min(max_len, max_length) - len(seq)))

        padded_sequences = [pad_sequence(seq, max_len) for seq in batch]
        return torch.stack(padded_sequences)

    def random_dic(self, dicts):
        dict_key_ls = list(dicts.keys())
        random.shuffle(dict_key_ls)
        new_dic = {}
        for key in dict_key_ls:
            new_dic[key] = dicts.get(key)
        return new_dic


class MELDReader(Reader):

    def __init__(self, model_path, data_dir, logger, window_size, batch_size, audio_encoder, has_dev=False) -> None:
        super().__init__(model_path, data_dir, logger, window_size, batch_size, has_dev, audio_encoder)

    '''
    [[{'user': , 'resp': }], [{'user': , 'resp': }]]
    '''
    def encode_data(self, data_type):
        data = load_json(os.path.join(self.data_dir, "{}_data.json".format(data_type)))
        audio_features = load_pickle(self.audio_feature_path)[data_type]
        video_features = load_pickle(self.video_feature_path)[data_type]
        encoded_data = {}

        for turn_num in data:
            if turn_num not in encoded_data:
                encoded_data[turn_num] = []
            for dialogue_ID in data[turn_num]:
                dialogues = data[turn_num][dialogue_ID]
                encoded_dial = []
                history = []
                audio_history = []
                video_history = []
                user_rec = []
                emotion_forecast_history = []
                user_history_ids = {}
                for Utterance_ID in dialogues:
                    enc = {}
                    utt = dialogues[Utterance_ID]
                    if utt['speaker'] not in user_history_ids:
                        user_history_ids[utt['speaker']] = []
                    enc["sr_no"] = utt["sr_no"]
                    enc["user"] = self.encode_text(utt["utterance"], self.bos_token, self.eos_token)
                    enc["history"] = []
                    enc["audio_history"] = []
                    enc["video_history"] = []
                    pre_users = []
                    enc["dia_emotion_forecast"] = []
                    enc["user_emotion_forecast"] = [self.tokenizer.encode('none')[0]]
                    if self.window_size > 0:
                        start = -self.window_size
                        for u in history[start:]:
                            enc["history"].append(u)
                        for a in audio_history[start:]:
                            enc["audio_history"].append(a)
                        for v in video_history[start:]:
                            enc["video_history"].append(v)
                        for u in user_rec[start:]:
                            pre_users.append(u)
                        for e in emotion_forecast_history[start:]:
                            enc["dia_emotion_forecast"].append(e)
                    enc["user_emotion_forecast"].append(self.tokenizer.encode(self.eos_status)[0])
                    enc["user_emotion_forecast"].append(self.tokenizer.encode(self.bos_resp)[0])
                    enc['resp'] = self.encode_text(utt["sentiment_label"], self.bos_token, self.eos_token)
                    enc['emotion_label'] = utt['emotion_label']
                    bos_emotion_index = enc['resp'].index(self.tokenizer.encode(self.bos_emotion)[0])
                    bos_sentiment_index = enc['resp'].index(self.tokenizer.encode(self.bos_sentiment)[0])

                    enc['emotion_token_index'] = bos_emotion_index + 1
                    enc['sentiment_token_index'] = bos_sentiment_index + 1
                    enc["history"].append(enc['user'])

                    emotion = self.encode_text(utt['speaker'] + ": " + utt['emotion'] + ' & ' + utt['sentiment'], self.bos_token, self.eos_token)
                    enc["dia_emotion_forecast"].append(emotion)

                    feature_key = utt["feature_key"]
                    fix = ('_' + self.audio_encoder) if 'hubert' in self.audio_encoder else ''
                    if feature_key + fix in audio_features:
                        enc['audio_feature'] = audio_features[feature_key + fix]
                    if feature_key in video_features:
                        enc['video_feature'] = video_features[feature_key]
                    enc['audio_history'].append(enc['audio_feature'])      #全部的audio特征，包括当前轮次
                    enc['video_history'].append(enc['video_feature'])
                    enc['speaker'] = utt['speaker']
                    pre_users.append(utt['speaker'])

                    node_num = len(enc['audio_history'])
                    g = [[0] * node_num for i in range(node_num)]
                    for i in range(node_num):
                        if i > 0:
                            g[i][i - 1] = 1 + np.exp(-1)
                        for j in range(i):
                            if pre_users[j] == pre_users[i]:
                                g[i][j] = 1 + np.exp(j - i)
                    enc['g'] = g

                    history.append(enc["user"][:-1] + enc['resp'][11:])
                    emotion_forecast_history.append(emotion)
                    user_history_ids[utt['speaker']].append(len(history) - 1)
                    audio_history.append(enc['audio_feature'])
                    video_history.append(enc['video_feature'])
                    encoded_dial.append(enc)
                    user_rec.append(utt['speaker'])
                encoded_data[turn_num].append(encoded_dial)
        
        return encoded_data

    def get_data_iterator(self, data, shuffle=False, dia_order=True, batch_size=1):
        if not dia_order:
            data_set = []
            for turn_num in data:
                dias = data[turn_num]
                for turns in dias:
                    for turn in turns:
                        data_set.append(turn)
            if shuffle:
                random.shuffle(data_set)
            i = 0
            while i < len(data_set):
                batch_dias = data_set[i : i + batch_size]
                batch = []
                for b_i in range(len(batch_dias)):
                    inputs = {}
                    inputs['label_ids'] = batch_dias[b_i]['resp']
                    inputs['class_label'] = batch_dias[b_i]['emotion_label']
                    inputs['sr_no'] = batch_dias[b_i]['sr_no']
                    inputs['emotion_token_index'] = batch_dias[b_i]['emotion_token_index']
                    inputs['sentiment_token_index'] = batch_dias[b_i]['sentiment_token_index']
                    inputs['dia_emotion_forecast'] = batch_dias[b_i]['dia_emotion_forecast']
                    inputs['user_emotion_forecast'] = batch_dias[b_i]['user_emotion_forecast']
                    inputs['speaker'] = batch_dias[b_i]['speaker']
                    inputs['video_history'] = batch_dias[b_i]['video_history']
                    inputs['audio_history'] = batch_dias[b_i]['audio_history']
                    inputs['text_history'] = batch_dias[b_i]['history']
                    inputs['g'] = batch_dias[b_i]['g']
                    batch.append(inputs)

                yield batch, False

                i += batch_size
        else:
            if shuffle:
                data = self.random_dic(data)
            for turn_num in data:
                dias = data[turn_num]
                if shuffle:
                    random.shuffle(dias)
                i = 0
                is_start = True
                while i < len(dias):
                    is_start = True
                    batch_dias = dias[i : i + batch_size]
                    
                    for t_i in range(int(turn_num)):
                        batch = []
                        for b_i in range(len(batch_dias)):
                            inputs = {}
                            inputs['label_ids'] = batch_dias[b_i][t_i]['resp']
                            inputs['class_label'] = batch_dias[b_i][t_i]['emotion_label']
                            inputs['sr_no'] = batch_dias[b_i][t_i]['sr_no']
                            inputs['emotion_token_index'] = batch_dias[b_i][t_i]['emotion_token_index']
                            inputs['sentiment_token_index'] = batch_dias[b_i][t_i]['sentiment_token_index']
                            inputs['dia_emotion_forecast'] = batch_dias[b_i][t_i]['dia_emotion_forecast']
                            inputs['user_emotion_forecast'] = batch_dias[b_i][t_i]['user_emotion_forecast']
                            inputs['speaker'] = batch_dias[b_i][t_i]['speaker']
                            inputs['audio_history'] = batch_dias[b_i][t_i]['audio_history']
                            inputs['video_history'] = batch_dias[b_i][t_i]['video_history']
                            inputs['text_history'] = batch_dias[b_i][t_i]['history']
                            inputs['g'] = batch_dias[b_i][t_i]['g']
                            batch.append(inputs)

                        yield batch, is_start
                        is_start = False

                    i += batch_size

class IEMOCAPReader(Reader):

    def __init__(self, model_path, data_dir, logger, window_size, batch_size, audio_encoder, has_dev=False) -> None:
        super().__init__(model_path, data_dir, logger, window_size, batch_size, has_dev, audio_encoder)

    def encode_data(self, data_type):
        data = load_json(os.path.join(self.data_dir, "{}_data.json".format(data_type)))
        audio_features = load_pickle(self.audio_feature_path)[data_type]
        video_features = load_pickle(self.video_feature_path)[data_type]
        encoded_data = {}

        for turn_num in data:
            if turn_num not in encoded_data:
                encoded_data[turn_num] = []
            for dialogue_ID in data[turn_num]:
                dialogues = data[turn_num][dialogue_ID]
                encoded_dial = []
                history = []
                audio_history = []
                video_history = []
                user_rec = []
                emotion_forecast_history = []
                user_history_ids = {}
                for Utterance_ID in dialogues:
                    enc = {}
                    utt = dialogues[Utterance_ID]
                    if utt['speaker'] not in user_history_ids:
                        user_history_ids[utt['speaker']] = []
                    enc["sr_no"] = utt["sr_no"]
                    enc["user"] = self.encode_text(utt["utterance"], self.bos_token, self.eos_token)
                    enc["history"] = []
                    enc["audio_history"] = []
                    enc["video_history"] = []
                    enc["dia_emotion_forecast"] = []
                    pre_users = []
                    enc["user_emotion_forecast"] = [self.tokenizer.encode('none')[0]]
                    if self.window_size > 0:
                        start = -self.window_size
                        for u in history[start:]:
                            enc["history"].append(u)
                        for a in audio_history[start:]:
                            enc["audio_history"].append(a)
                        for v in video_history[start:]:
                            enc["video_history"].append(v)
                        for u in user_rec[start:]:
                            pre_users.append(u)
                        for e in emotion_forecast_history[start:]:
                            enc["dia_emotion_forecast"].append(e)
                    enc["user_emotion_forecast"].append(self.tokenizer.encode(self.eos_status)[0])
                    enc["user_emotion_forecast"].append(self.tokenizer.encode(self.bos_resp)[0])
                    enc['resp'] = self.encode_text(utt["sentiment_label"], self.bos_token, self.eos_token)
                    enc['emotion_label'] = utt['emotion_label']
                    bos_emotion_index = enc['resp'].index(self.tokenizer.encode(self.bos_emotion)[0])
                    bos_sentiment_index = enc['resp'].index(self.tokenizer.encode(self.bos_sentiment)[0])

                    enc['emotion_token_index'] = bos_emotion_index + 1
                    enc['sentiment_token_index'] = bos_sentiment_index + 1
                    enc["history"].append(enc['user'])
                    enc["dia_emotion_forecast"].append(self.encode_text(utt['speaker'] + ": undetermined & undetermined", self.bos_token, self.eos_token))

                    emotion = self.encode_text(utt['speaker'] + ": " + utt['emotion'] + ' & ' + utt['sentiment'], self.bos_token, self.eos_token)

                    feature_key = utt["feature_key"]
                    fix = ('_' + self.audio_encoder) if 'hubert' in self.audio_encoder else ''
                    enc['audio_feature'] = audio_features[feature_key + fix]
                    enc['video_feature'] = video_features[feature_key]
                    enc['audio_history'].append(enc['audio_feature'])      #全部的audio特征，包括当前轮次
                    enc['video_history'].append(enc['video_feature'])
                    enc['speaker'] = utt['speaker']
                    pre_users.append(utt['speaker'])

                    node_num = len(enc['audio_history'])
                    g = [[0] * node_num for i in range(node_num)]
                    for i in range(node_num):
                        if i > 0:
                            g[i][i - 1] = 1 + np.exp(-1)
                        for j in range(i):
                            if pre_users[j] == pre_users[i]:
                                g[i][j] = 1 + np.exp(j - i)
                    enc['g'] = g

                    history.append(enc["user"][:-1] + enc['resp'][11:])
                    emotion_forecast_history.append(emotion)
                    user_history_ids[utt['speaker']].append(len(history) - 1)
                    audio_history.append(enc['audio_feature'])
                    video_history.append(enc['video_feature'])
                    encoded_dial.append(enc)
                    user_rec.append(utt['speaker'])
                encoded_data[turn_num].append(encoded_dial)
        
        return encoded_data

    def get_data_iterator(self, data, shuffle=False, dia_order=True, batch_size=1):
        if not dia_order:
            data_set = []
            for turn_num in data:
                dias = data[turn_num]
                for turns in dias:
                    for turn in turns:
                        data_set.append(turn)
            if shuffle:
                random.shuffle(data_set)
            i = 0
            while i < len(data_set):
                batch_dias = data_set[i : i + batch_size]
                batch = []
                for b_i in range(len(batch_dias)):
                    inputs = {}
                    inputs['label_ids'] = batch_dias[b_i]['resp']
                    inputs['class_label'] = batch_dias[b_i]['emotion_label']
                    inputs['sr_no'] = batch_dias[b_i]['sr_no']
                    inputs['emotion_token_index'] = batch_dias[b_i]['emotion_token_index']
                    inputs['sentiment_token_index'] = batch_dias[b_i]['sentiment_token_index']
                    inputs['dia_emotion_forecast'] = batch_dias[b_i]['dia_emotion_forecast']
                    inputs['user_emotion_forecast'] = batch_dias[b_i]['user_emotion_forecast']
                    inputs['speaker'] = batch_dias[b_i]['speaker']
                    inputs['video_history'] = batch_dias[b_i]['video_history']
                    inputs['audio_history'] = batch_dias[b_i]['audio_history']
                    inputs['text_history'] = batch_dias[b_i]['history']
                    inputs['g'] = batch_dias[b_i]['g']
                    batch.append(inputs)

                yield batch, False

                i += batch_size
        else:
            if shuffle:
                data = self.random_dic(data)
            for turn_num in data:
                dias = data[turn_num]
                if shuffle:
                    random.shuffle(dias)
                i = 0
                is_start = True
                while i < len(dias):
                    is_start = True
                    batch_dias = dias[i : i + batch_size]
                    
                    for t_i in range(int(turn_num)):
                        batch = []
                        for b_i in range(len(batch_dias)):
                            inputs = {}
                            inputs['label_ids'] = batch_dias[b_i][t_i]['resp']
                            inputs['class_label'] = batch_dias[b_i][t_i]['emotion_label']
                            inputs['sr_no'] = batch_dias[b_i][t_i]['sr_no']
                            inputs['emotion_token_index'] = batch_dias[b_i][t_i]['emotion_token_index']
                            inputs['sentiment_token_index'] = batch_dias[b_i][t_i]['sentiment_token_index']
                            inputs['dia_emotion_forecast'] = batch_dias[b_i][t_i]['dia_emotion_forecast']
                            inputs['user_emotion_forecast'] = batch_dias[b_i][t_i]['user_emotion_forecast']
                            inputs['speaker'] = batch_dias[b_i][t_i]['speaker']
                            inputs['audio_history'] = batch_dias[b_i][t_i]['audio_history']
                            inputs['video_history'] = batch_dias[b_i][t_i]['video_history']
                            inputs['text_history'] = batch_dias[b_i][t_i]['history']
                            inputs['g'] = batch_dias[b_i][t_i]['g']
                            batch.append(inputs)

                        yield batch, is_start
                        is_start = False

                    i += batch_size
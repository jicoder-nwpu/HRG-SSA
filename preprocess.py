import os
import json
import pandas as pd
import math
from io_utils import save_json, load_json
import csv
import pickle
import numpy as np
import random

class MeldPreprocess():

    def __init__(self, dataset_path, structured_label, emotion_first) -> None:
        self.dataset_path = dataset_path
        self.structured_label = structured_label
        self.emotion_first = emotion_first
        self.train_path = dataset_path + '/train_splits'
        self.dev_path = dataset_path + '/dev_splits_complete'
        self.test_path = dataset_path + '/output_repeated_splits_test'
        self.train_sent_emo = dataset_path + '/train_sent_emo.csv'
        self.dev_sent_emo = dataset_path + '/dev_sent_emo.csv'
        self.test_sent_emo = dataset_path + '/test_sent_emo.csv'
        self.train_wav_path = self.train_path + '/wavs'
        self.dev_wav_path = self.dev_path + '/wavs'
        self.test_wav_path = self.test_path + '/wavs'
        self.train_json = './MELD/train_data.json'
        self.dev_json = './MELD/dev_data.json'
        self.test_json = './MELD/test_data.json'
        self.except_utts = {}
        self.except_utts['train'] = ['dia125_utt3']
        self.except_utts['dev'] = ['dia110_utt7']
        self.except_utts['test'] = []
        self.emotion2id = {
                        'anger' : 0, 
                        'surprise': 1, 
                        'disgust': 2, 
                        'fear': 3, 
                        'neutral': 4, 
                        'sadness': 5, 
                        'joy': 6
                    }
        
    def mp4_2_wav(self):
        if not os.path.exists(self.train_wav_path):
            os.makedirs(self.train_wav_path)
            for file in os.listdir(self.train_path):
                command = self.ffmpeg_cmd(self.train_path, self.train_wav_path, file)
                os.system(command)

        if not os.path.exists(self.dev_wav_path):
            os.makedirs(self.dev_wav_path)
            for file in os.listdir(self.dev_path):
                command = self.ffmpeg_cmd(self.dev_path, self.dev_wav_path, file)
                print(command)
                os.system(command)

        if not os.path.exists(self.test_wav_path):
            os.makedirs(self.test_wav_path)
            for file in os.listdir(self.test_path):
                command = self.ffmpeg_cmd(self.test_path, self.test_wav_path, file)
                os.system(command)
        
    '''
    首先安装ffmpeg
    sudo apt update
    sudo apt install ffmpeg
    '''
    def ffmpeg_cmd(self, pa_path, root_path, file):
        filename = file.split('.')[0] + '.wav'
        filename = os.path.join(root_path, filename)
        return 'ffmpeg -i ' + os.path.join(pa_path, file) + ' -ac 1 -ar 16000 -sample_fmt s16 ' +  filename

    def clean_data(self, sent):
        sent = sent.replace("\u0092", "'")
        sent = sent.replace("\u0085", " ")
        sent = sent.replace("\u0093", "")
        sent = sent.replace("\u0094", "")
        sent = sent.replace("\u00a0", "")
        sent = sent.replace("\u0097", " ")
        sent = sent.replace("\u0091", "'")
        sent = sent.replace("\"", "")
        return sent
    
    def get_label(self, speaker, emotion, sentiment, structured_label, emotion_first):
        if structured_label:
            return "<bos_emotion> " + emotion + " <eos_emotion>, <bos_sentiment> " + sentiment + " <eos_sentiment>."
        if emotion_first:
            return "In the last round of the above dialogue, the speaker " + speaker + "'s emotion is <bos_emotion> " + emotion + " <eos_emotion>, and the sentiment is <bos_sentiment> " + sentiment + " <eos_sentiment>."
        return "In the last round of the above dialogue, the speaker " + speaker + "'s sentiment is <bos_sentiment> " + sentiment + " <eos_sentiment>, and the emotion is <bos_emotion> " + emotion + " <eos_emotion>."

    def csv_2_json(self, json_file, sent_emo, data_type):
        # if not os.path.exists(json_file):
        data = {}
        with open(sent_emo, 'r') as fin:
            reader = fin.readlines()
            for row in reader:
                row = self.clean_data(row.replace('\n', ''))
                if 'Utterance_ID' in row:
                    continue
                words = row.split(',')
                dialogue_ID = words[-8]
                Utterance_ID = words[-7]
                if ('dia' + str(dialogue_ID) + '_utt' + str(Utterance_ID)) in self.except_utts[data_type]:
                    continue
                Sr_No = words[0]
                Utterance = ','.join(words[1:-11])
                Speaker = words[-11]
                Emotion = words[-10]
                Sentiment = words[-9]
                if dialogue_ID not in data:
                    data[dialogue_ID] = {}
                data[dialogue_ID][Utterance_ID] = {}
                data[dialogue_ID][Utterance_ID]['Sr_No'] =  Sr_No
                data[dialogue_ID][Utterance_ID]['Utterance'] = Speaker + ": " + Utterance
                data[dialogue_ID][Utterance_ID]['Speaker'] =  Speaker
                data[dialogue_ID][Utterance_ID]['Emotion'] =  Emotion
                data[dialogue_ID][Utterance_ID]['Sentiment'] =  Sentiment
                data[dialogue_ID][Utterance_ID]['Sentiment_Label'] =  self.get_label(Speaker, Emotion, Sentiment, self.structured_label, self.emotion_first)
                data[dialogue_ID][Utterance_ID]['Emotion_Label'] =  self.emotion2id[Emotion]
                data[dialogue_ID][Utterance_ID]['mp4_file'] = 'dia' + str(dialogue_ID) + '_utt' + str(Utterance_ID) + '.mp4'
                data[dialogue_ID][Utterance_ID]['wav_file'] = 'dia' + str(dialogue_ID) + '_utt' + str(Utterance_ID) + '.wav'
                data[dialogue_ID][Utterance_ID]['feature_key'] = 'dia' + str(dialogue_ID) + '_utt' + str(Utterance_ID)

        save_data = {}
        for dia in data:
            turn_num = len(data[dia])
            if turn_num not in save_data:
                save_data[turn_num] = {}
            save_data[turn_num][dia] = data[dia]
        with open(json_file, 'w') as fout:
            json.dump(save_data, fout, indent=4)

    def get_jsons(self):
        self.csv_2_json(self.train_json, self.train_sent_emo, 'train')
        self.csv_2_json(self.dev_json, self.dev_sent_emo, 'dev')
        self.csv_2_json(self.test_json, self.test_sent_emo, 'test')


class IEMOCAPPreprocess():

    def __init__(self, dataset_path, structured_label, emotion_first) -> None:
        self.structured_label = structured_label
        self.emotion_first = emotion_first
        self.sentiment_2_label = {'sad': 'negative', 'fru': 'negative', 'neu': 'neutral', 'ang': 'negative', 'exc': 'positive', 'hap': 'positive'}
        self.emotion_2_label = {'sad': 'sadness', 'fru': 'frustration', 'neu': 'neutral', 'ang': 'anger', 'exc': 'excitement', 'hap': 'happiness'}
        self.emotion_2_id = {'sad': 0, 'fru': 1, 'neu': 2, 'ang': 3, 'exc': 4, 'hap': 5}
        self.role_2_label = {'m': 'male', 'f': 'female'}
        self.train_path = dataset_path + '/train_data_origin.json'
        self.test_path = dataset_path + '/test_data_origin.json'
        self.train_json_path = dataset_path + '/train_data.json'
        self.test_json_path = dataset_path + '/test_data.json'

        self.train_data = {}
        self.test_data = {}

    def read_data(self, ):
        rows = load_json(self.train_path)
        for dia_id in rows:
            turn_num = len(rows[dia_id])
            if turn_num not in self.train_data:
                self.train_data[turn_num] = {}
            self.train_data[turn_num][dia_id] = {}
            for u_id in rows[dia_id]:
                rows[dia_id][u_id]['sr_no'] = u_id
                rows[dia_id][u_id]['feature_key'] = u_id
                rows[dia_id][u_id]['utterance'] = self.role_2_label[rows[dia_id][u_id]['role']] + ': ' + rows[dia_id][u_id]['text']
                rows[dia_id][u_id]['speaker'] =  self.role_2_label[rows[dia_id][u_id]['role']]
                rows[dia_id][u_id]['emotion'] =  self.emotion_2_label[rows[dia_id][u_id]['label']]
                rows[dia_id][u_id]['sentiment'] =  self.sentiment_2_label[rows[dia_id][u_id]['label']]
                rows[dia_id][u_id]['sentiment_label'] =  self.get_label(self.role_2_label[rows[dia_id][u_id]['role']], rows[dia_id][u_id]['emotion'], rows[dia_id][u_id]['sentiment'], self.structured_label, self.emotion_first)
                rows[dia_id][u_id]['emotion_label'] =  self.emotion_2_id[rows[dia_id][u_id]['label']]
                self.train_data[turn_num][dia_id] = rows[dia_id]
        save_json(self.train_data, self.train_json_path)

        rows = load_json(self.test_path)
        for dia_id in rows:
            turn_num = len(rows[dia_id])
            if turn_num not in self.test_data:
                self.test_data[turn_num] = {}
            self.test_data[turn_num][dia_id] = {}
            for u_id in rows[dia_id]:
                rows[dia_id][u_id]['sr_no'] = u_id
                rows[dia_id][u_id]['feature_key'] = u_id
                rows[dia_id][u_id]['utterance'] = self.role_2_label[rows[dia_id][u_id]['role']] + ': ' + rows[dia_id][u_id]['text']
                rows[dia_id][u_id]['speaker'] =  self.role_2_label[rows[dia_id][u_id]['role']]
                rows[dia_id][u_id]['emotion'] =  self.emotion_2_label[rows[dia_id][u_id]['label']]
                rows[dia_id][u_id]['sentiment'] =  self.sentiment_2_label[rows[dia_id][u_id]['label']]
                rows[dia_id][u_id]['sentiment_label'] =  self.get_label(self.role_2_label[rows[dia_id][u_id]['role']], rows[dia_id][u_id]['emotion'], rows[dia_id][u_id]['sentiment'], self.structured_label, self.emotion_first)
                rows[dia_id][u_id]['emotion_label'] =  self.emotion_2_id[rows[dia_id][u_id]['label']]
                self.test_data[turn_num][dia_id] = rows[dia_id]
        save_json(self.test_data, self.test_json_path)
 
    def get_label(self, speaker, emotion, sentiment, structured_label, emotion_first):
        if structured_label:
            return "<bos_sentiment> " + sentiment + " <eos_sentiment>, <bos_emotion> " + emotion + " <eos_emotion>."
        if emotion_first:
            return "In the last round of the above dialogue, the " + speaker + " speaker's emotion is <bos_emotion> " + emotion + " <eos_emotion>, and the sentiment is <bos_sentiment> " + sentiment + " <eos_sentiment>."
        return "In the last round of the above dialogue, the " + speaker + " speaker's sentiment is <bos_sentiment> " + sentiment + " <eos_sentiment>."


if __name__ == '__main__':
    # preprocessor = MeldPreprocess('/sda/jihongru/gat/MELD')
    # preprocessor.mp4_2_wav()
    # preprocessor.get_jsons()
    iemocap_processer = IEMOCAPPreprocess('./IEMOCAP', False, True)
    iemocap_processer.read_data()
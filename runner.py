from model import T5ForConditionalGeneration
import os
import glob
import shutil
import torch
import torch.nn.functional as F
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule
from io_utils import *
from reader import *
from torch.utils.data import DataLoader
from reporter import Reporter
import time
import copy, numpy
from sklearn import metrics
from labels_map import evaluate_metrics
import scipy.sparse as sp

class BaseRunner():
    def __init__(self, cfg):
        self.logger = get_or_create_logger(__name__, cfg.model_dir)
        self.cfg = cfg
        if self.cfg.ckpt is not None:
            self.model_path = self.cfg.ckpt
        else:
            self.model_path = self.cfg.backbone
        if self.cfg.dataset == 'meld':
            self.reader = MELDReader(self.model_path, self.cfg.data_dir, self.logger, cfg.window_size, self.cfg.batch_size, self.cfg.audio_encoder)
        elif self.cfg.dataset == 'iemocap':
            self.reader = IEMOCAPReader(self.model_path, self.cfg.data_dir, self.logger, cfg.window_size, self.cfg.batch_size, self.cfg.audio_encoder)
        self.model = self.load_model()
        self.max_length = self.model.config.n_positions
        self.pad_token = self.reader.pad_token_id

    def load_model(self, ckpt=None):
        self.logger.info("Load model from {}".format(self.model_path if ckpt is None else ckpt))

        model_wrapper = T5ForConditionalGeneration

        model = model_wrapper.from_pretrained(self.model_path if ckpt is None else ckpt, ignore_mismatched_sizes=True)

        model.resize_token_embeddings(self.reader.vocab_size)

        t5_state_dict=torch.load('./pretrained_model/pytorch_model.bin')
        for n,p in model.decoder.named_parameters():
            if "layer.4" in n:
                name_split = n.split('layer.4')
                name = 'decoder.' + name_split[0] + 'layer.1' + name_split[1]
                p.data.copy_(t5_state_dict[name].data)

        model.to(self.cfg.device)

        return model

    def save_model(self, epoch):
        latest_ckpt = "ckpt-epoch{}".format(epoch)
        save_path = os.path.join(self.cfg.model_dir, latest_ckpt)

        model = self.model

        model.save_pretrained(save_path)

        checkpoints = sorted(
            glob.glob(os.path.join(self.cfg.model_dir, "ckpt-*")),
            key=os.path.getmtime,
            reverse=True)

        checkpoints_to_be_deleted = checkpoints[self.cfg.max_to_keep_ckpt:]

        for ckpt in checkpoints_to_be_deleted:
            shutil.rmtree(ckpt)

        return save_path

    def get_optimizer_and_scheduler(self, num_traininig_steps_per_epoch, learning_rate, epochs, warmup_ratio, is_freeze=False):
        num_train_steps = (num_traininig_steps_per_epoch *
            epochs) // self.cfg.grad_accum_steps

        if self.cfg.warmup_steps >= 0:
            num_warmup_steps = self.cfg.warmup_steps
        else:
            num_warmup_steps = int(num_train_steps * warmup_ratio)

        self.logger.info("Total training steps = {}, warmup steps = {}".format(
            num_train_steps, num_warmup_steps))
        
        if is_freeze:
            for name, param in self.model.named_parameters():
                if 'layer.4' not in name and 'layer.3' not in name and '_encoder' not in name and 'gat' not in name and '_mapping' not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        else:
            for name, param in self.model.named_parameters():
                param.requires_grad = True

        optimizer = AdamW(filter(lambda p : p.requires_grad, self.model.parameters()), lr=learning_rate)

        if self.cfg.no_learning_rate_decay:
            scheduler = get_constant_schedule(optimizer)
        else:
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_train_steps)

        return optimizer, scheduler

    def count_tokens(self, pred, label, pad_id):
        pred = pred.view(-1)
        label = label.view(-1)

        num_count = label.ne(pad_id).long().sum()
        num_correct = torch.eq(pred, label).long().sum()

        return num_correct, num_count

    def count_spans(self, pred, label):
        pred = pred.view(-1, 2)

        num_count = label.ne(-1).long().sum()
        num_correct = torch.eq(pred, label).long().sum()

        return num_correct, num_count

    def add_eos(self, features, is_audio):
        if is_audio:
            eos_id = torch.tensor([self.reader.tokenizer.encode(self.reader.eos_audio)[0]]).to(self.cfg.device)
        else:
            eos_id = torch.tensor([self.reader.tokenizer.encode(self.reader.eos_video)[0]]).to(self.cfg.device)
        eos_embedding = self.model.shared(eos_id)
        for i in range(len(features)):
            if isinstance(features[i], numpy.ndarray) or features[i].device.type == 'cpu':
                features[i] = torch.tensor(features[i], dtype=torch.float).to(self.cfg.device)
            features[i] = torch.cat((features[i], eos_embedding))
        return features
    
    def add_bos_eos(self, features, is_audio):
        if is_audio:
            eos_id = torch.tensor([self.reader.tokenizer.encode(self.reader.eos_audio)[0]]).to(self.cfg.device)
            bos_id = torch.tensor([self.reader.tokenizer.encode(self.reader.bos_audio)[0]]).to(self.cfg.device)
        else:
            eos_id = torch.tensor([self.reader.tokenizer.encode(self.reader.eos_video)[0]]).to(self.cfg.device)
            bos_id = torch.tensor([self.reader.tokenizer.encode(self.reader.bos_video)[0]]).to(self.cfg.device)
        eos = self.model.shared(eos_id)
        bos = self.model.shared(bos_id)
        bos_embedding = []
        eos_embedding = []
        for i in range(len(features)):
            bos_embedding.append(bos)
            eos_embedding.append(eos)
        bos_embedding = torch.stack(bos_embedding)
        eos_embedding = torch.stack(eos_embedding)
        features = torch.cat((features, eos_embedding), dim=1)
        features = torch.cat((bos_embedding, features), dim=1)
        return features
    
    def pad_features_pre(self, features):
        max_len = 0
        for i in range(len(features)):
            if len(features[i]) > self.max_length:
                features[i] = features[i][-self.max_length:]
            max_len = max(max_len, len(features[i]))
        for i in range(len(features)):
            if max_len - len(features[i]) == 0 :
                continue
            pad_ids = torch.tensor([self.reader.pad_token_id] * (max_len - len(features[i]))).to(self.cfg.device)
            pad_embedding = self.model.shared(pad_ids)
            features[i] = torch.cat((pad_embedding, features[i]))
        return torch.stack(features)

    def pad_features(self, features, ty):
        max_len = 0
        masks = []
        audio_prefix_ids = torch.tensor([self.reader.tokenizer.encode(self.reader.bos_audio)[0]]).to(self.cfg.device)
        audio_prefix = self.model.shared(audio_prefix_ids)
        video_prefix_ids = torch.tensor([self.reader.tokenizer.encode(self.reader.bos_video)[0]]).to(self.cfg.device)
        video_prefix = self.model.shared(video_prefix_ids)
        pre_len = 0 if ty == 'text' else len(audio_prefix)
        eos_index = []
        for i in range(len(features)):
            if len(features[i]) + pre_len > self.max_length:
                features[i] = features[i][-self.max_length + pre_len:]
            if ty == 'audio':
                features[i] = torch.cat((audio_prefix, features[i]))
            elif ty == 'video':
                features[i] = torch.cat((video_prefix, features[i]))
            max_len = max(max_len, len(features[i]))
            masks.append([1] * len(features[i]))
            eos_index.append(len(features[i]) - 1)
        for i in range(len(features)):
            if max_len - len(features[i]) == 0 :
                continue
            masks[i] += [0] * (max_len - len(masks[i]))
            pad_ids = torch.tensor([self.reader.pad_token_id] * (max_len - len(features[i]))).to(self.cfg.device)
            pad_embedding = self.model.shared(pad_ids)
            features[i] = torch.cat((features[i], pad_embedding))
        masks = torch.tensor(masks).to(self.cfg.device)
        return torch.stack(features), masks, eos_index
    
    def pad_sequence(self, seq, max_len):
        if len(seq) > self.max_length:
            return torch.tensor([3] + list(seq)[-self.max_length + 1:])
        return torch.tensor(list(seq) + [self.pad_token]*(min(max_len, self.max_length) - len(seq)))
    
    def pad_batch_sequences(self, sequences):
        max_len = 0
        for seq in sequences:
            max_len = max(max_len, len(seq))
        return [self.pad_sequence(seq, max_len) for seq in sequences]
    
    def pad_sequence_pre(self, seq, max_len):
        if len(seq) > self.max_length:
            return torch.tensor([3] + list(seq)[-self.max_length + 1:])
        return torch.tensor([self.pad_token]*(min(max_len, self.max_length) - len(seq)) + list(seq))
    
    def pad_batch_sequences_pre(self, sequences):
        max_len = 0
        for seq in sequences:
            max_len = max(max_len, len(seq))
        return [self.pad_sequence_pre(seq, max_len) for seq in sequences]
    
    def process_batch(self, batch, predict_history=None, predict_emotion_history=None):
        batch = batch[0]
        batch_size = len(batch)
        batch_adj_matrix = []
        labels_ids = []
        batch_input_ids = []
        sr_nos = []
        sentiment_token_indexs = []
        emotion_token_indexs = []
        batch_audio_features = None
        batch_video_features = None
        batch_text_masks = None
        batch_audio_edges = None
        batch_video_edges = None
        class_labels = []
        batch_dia_emotion_forecast = []
        batch_dia_emotion_forecast_edges = []
        batch_text_node_features = []
        batch_text_edges = []
        emotion_text = []
        text_text = []

        node_num = 0
        for i in range(batch_size):
            batch_dia_emotion_forecast.append([])
            batch_text_node_features.append([])
            if predict_history is None:
                text_history = batch[i]['text_history']
            else:
                text_history = batch[i]['text_history']
                if len(text_history) > 1:
                    text_history[-2] = predict_history[i][-1]
            if predict_emotion_history is None:
                emotion_history = batch[i]['dia_emotion_forecast']
            else:
                emotion_history = batch[i]['dia_emotion_forecast']
                if len(emotion_history) > 1:
                    emotion_history[-2] = predict_emotion_history[i][-1]
            emotion_text.append(emotion_history)
            text_text.append(text_history)
            g = batch[i]['g']
            g = torch.tensor(g, dtype=torch.float)
            g = torch.where(g == 0, float("-inf"), g)
            g = F.softmax(g, dim=1)
            labels_ids.append(batch[i]['label_ids'])
            class_labels.append(batch[i]['class_label'])
            sr_nos.append(batch[i]['sr_no'])
            sentiment_token_indexs.append(int(batch[i]['sentiment_token_index']))
            emotion_token_indexs.append(int(batch[i]['emotion_token_index']))

            input_ids = []
            for text in text_history:
                input_ids += text
            batch_input_ids.append(input_ids)

            edges = []
            for j in range(len(batch[i]['video_history'])):
                edges.append([j, j])
                for k in range(j):
                    if batch[i]['g'][j][k] > 0:
                        edges.append([k, j])
            batch_adj_matrix.append(edges)
            node_num = max(node_num, len(batch[i]['video_history']))

        batch_input_ids = torch.stack(self.pad_batch_sequences(batch_input_ids))
        batch_text_masks =  torch.where(batch_input_ids == self.reader.pad_token_id, 0, 1)
        batch_input_ids = batch_input_ids.to(self.cfg.device)
        batch_text_masks = batch_text_masks.to(self.cfg.device)
        batch_text_features = self.model(input_ids=batch_input_ids,
                                        attention_mask=batch_text_masks,
                                        encoder_only=True)

        
        for j in range(node_num):
            cur_text = []
            cur_index = []
            text_eos_indx = []
            for i in range(len(batch)):
                if j < len(batch[i]["dia_emotion_forecast"]):
                    cur_index.append(i)
                    cur_text.append(emotion_text[i][j])
                    text_eos_indx.append(len(emotion_text[i][j]) - 1)
            cur_input_ids = torch.stack(self.pad_batch_sequences(cur_text))
            cur_text_masks =  torch.where(cur_input_ids == self.reader.pad_token_id, 0, 1)
            cur_input_ids = cur_input_ids.to(self.cfg.device)
            cur_text_masks = cur_text_masks.to(self.cfg.device)
            cur_text_features = self.model(input_ids=cur_input_ids,
                                            attention_mask=cur_text_masks,
                                            encoder_only=True)[0]
            
            for i in range(len(cur_index)):
                batch_dia_emotion_forecast[cur_index[i]].append(cur_text_features[i][text_eos_indx[i]])
        for i in range(batch_size):
            batch_dia_emotion_forecast[i] = torch.stack(batch_dia_emotion_forecast[i])

            if not self.cfg.no_use_relation and len(batch[i]['dia_emotion_forecast']) > 2:
                edge_index1=torch.LongTensor(batch_adj_matrix[i]).transpose(0,1)
                edge_index1=edge_index1.cuda(device=self.cfg.device)
                row,col=edge_index1.cpu()
                cc=torch.ones(row.shape[0]).cpu()
                d=1-(sp.coo_matrix((cc, (row, col)), shape=(len(batch[i]['dia_emotion_forecast']), len(batch[i]['dia_emotion_forecast'])))).toarray()
                d[d<1]=0
                c = torch.from_numpy(d).cuda()
                emotion_edges = batch_adj_matrix[i] + self.model.emotion_edge_relation(batch_dia_emotion_forecast[i], c)
            else:
                emotion_edges = batch_adj_matrix[i]

            batch_dia_emotion_forecast_edges.append(torch.LongTensor(emotion_edges).transpose(0,1).to(self.cfg.device))
        
        past_states = []
        for j in range(node_num):
            cur_text = []
            cur_index = []
            text_eos_indx = []
            for i in range(len(batch)):
                if j < len(batch[i]['text_history']):
                    cur_index.append(i)
                    cur_text.append(text_text[i][j])
                    text_eos_indx.append(len(text_text[i][j]) - 1)
            cur_input_ids = torch.stack(self.pad_batch_sequences(cur_text))
            cur_text_masks =  torch.where(cur_input_ids == self.reader.pad_token_id, 0, 1)
            cur_input_ids = cur_input_ids.to(self.cfg.device)
            cur_text_masks = cur_text_masks.to(self.cfg.device)
            cur_text_features = self.model(input_ids=cur_input_ids,
                                            attention_mask=cur_text_masks,
                                            encoder_only=True)[0]
            
            past_states.append((cur_text_features, cur_text_masks, cur_index, text_eos_indx))
        
        audio_past_states = []
        if self.cfg.use_audio_mode:
            batch_audio_features = []
            for i in range(len(batch)):
                batch_audio_features.append([])
            for j in range(node_num):
                cur_text_features, cur_text_masks, cur_index, text_eos_indx = past_states[j]

                turn_audio_features = []
                for i in range(len(batch)):
                    if j < len(batch[i]['audio_history']):
                        if isinstance(batch[i]['audio_history'][j], numpy.ndarray):
                            batch[i]['audio_history'][j] = torch.tensor(batch[i]['audio_history'][j], device=self.cfg.device, dtype=self.model.dtype)
                        if batch[i]['audio_history'][j].device.type == 'cpu':
                            batch[i]['audio_history'][j] = batch[i]['audio_history'][j].to(self.cfg.device)
                        if len(batch[i]['audio_history'][j].shape) == 1:
                            batch[i]['audio_history'][j] = batch[i]['audio_history'][j].unsqueeze(dim=0)
                        audio_hidden_states = self.model.audio_mapping(batch[i]['audio_history'][j])
                        turn_audio_features.append(audio_hidden_states)
                turn_audio_features, turn_audio_masks, audio_eos_index = self.pad_features(turn_audio_features, 'text')

                turn_audio_features_h = self.model(inputs_embeds=turn_audio_features,
                                                decoder_attention_mask=turn_audio_masks,
                                                encoder_outputs=cur_text_features,
                                                attention_mask=cur_text_masks,
                                                audio_encoder_only=True)
                audio_past_states.append((turn_audio_features, turn_audio_masks, cur_index))
                for i in range(len(cur_index)):
                    batch_audio_features[cur_index[i]].append(turn_audio_features_h[i][audio_eos_index[i]])

            batch_audio_edges = []
            for i in range(len(batch)):
                batch_audio_features[i] = torch.stack(batch_audio_features[i])

                if not self.cfg.no_use_relation and len(batch[i]['audio_history']) > 2:
                    edge_index1=torch.LongTensor(batch_adj_matrix[i]).transpose(0,1)
                    edge_index1=edge_index1.cuda(device=self.cfg.device)
                    row,col=edge_index1.cpu()
                    cc=torch.ones(row.shape[0]).cpu()
                    d=1-(sp.coo_matrix((cc, (row, col)), shape=(len(batch[i]['audio_history']), len(batch[i]['audio_history'])))).toarray()
                    d[d<1]=0
                    c = torch.from_numpy(d).cuda()
                    audio_edges = batch_adj_matrix[i] + self.model.audio_edge_relation(batch_audio_features[i], c)
                else:
                    audio_edges = batch_adj_matrix[i]

                batch_audio_edges.append(torch.LongTensor(audio_edges).transpose(0,1).to(self.cfg.device))

        video_past_states = []
        if self.cfg.use_video_mode:
            batch_video_features = []
            for i in range(len(batch)):
                batch_video_features.append([])
            for j in range(node_num):
                cur_text_features, cur_text_masks, cur_index, text_eos_indx = past_states[j]

                turn_video_features = []
                for i in range(len(batch)):
                    if j < len(batch[i]['video_history']):
                        if isinstance(batch[i]['video_history'][j], numpy.ndarray):
                            batch[i]['video_history'][j] = torch.tensor(batch[i]['video_history'][j], dtype=self.model.dtype).to(self.cfg.device)
                        if batch[i]['video_history'][j].device.type == 'cpu':
                            batch[i]['video_history'][j] = batch[i]['video_history'][j].to(self.cfg.device)
                        if len(batch[i]['video_history'][j].shape) == 1:
                            batch[i]['video_history'][j] = batch[i]['video_history'][j].unsqueeze(dim=0)
                        video_hidden_states = self.model.video_mapping(batch[i]['video_history'][j])
                        turn_video_features.append(video_hidden_states)
                turn_video_features, turn_video_masks, video_eos_index = self.pad_features(turn_video_features, 'text')

                turn_video_features_h = self.model(inputs_embeds=turn_video_features,
                                                decoder_attention_mask=turn_video_masks,
                                                encoder_outputs=cur_text_features,
                                                attention_mask=cur_text_masks,
                                                video_encoder_only=True)
            
                video_past_states.append((turn_video_features, turn_video_masks, cur_index))
                for i in range(len(cur_index)):
                    batch_video_features[cur_index[i]].append(turn_video_features_h[i][video_eos_index[i]])

            batch_video_edges = []
            for i in range(len(batch)):
                batch_video_features[i] = torch.stack(batch_video_features[i])

                if not self.cfg.no_use_relation and len(batch[i]['video_history']) > 2:
                    edge_index1=torch.LongTensor(batch_adj_matrix[i]).transpose(0,1)
                    edge_index1=edge_index1.cuda(device=self.cfg.device)
                    row,col=edge_index1.cpu()
                    cc=torch.ones(row.shape[0]).cpu()
                    d=1-(sp.coo_matrix((cc, (row, col)), shape=(len(batch[i]['video_history']), len(batch[i]['video_history'])))).toarray()
                    d[d<1]=0
                    c = torch.from_numpy(d).cuda()
                    video_deges = batch_adj_matrix[i] + self.model.video_edge_relation(batch_video_features[i], c)
                else:
                    video_deges = batch_adj_matrix[i]

                batch_video_edges.append(torch.LongTensor(video_deges).transpose(0,1).to(self.cfg.device))
            
        for j in range(node_num):
            cur_text_features, cur_text_masks, cur_index, text_eos_indx = past_states[j]
            if self.cfg.use_audio_mode:
                turn_audio_features, turn_audio_masks, cur_index = audio_past_states[j]
            if self.cfg.use_video_mode:
                turn_video_features, turn_video_masks, cur_index = video_past_states[j]
            if self.cfg.use_audio_mode and self.cfg.use_video_mode:
                cur_text_features = self.model(inputs_embeds=cur_text_features,
                                            decoder_attention_mask=cur_text_masks,
                                            encoder_outputs=torch.cat((turn_audio_features, turn_video_features), dim=1).to(turn_audio_features.device),
                                            attention_mask=torch.cat((turn_audio_masks, turn_video_masks), dim=1).to(turn_audio_features.device),
                                            text_encoder_only=True)
            elif self.cfg.use_audio_mode:
                cur_text_features = self.model(inputs_embeds=cur_text_features,
                                            decoder_attention_mask=cur_text_masks,
                                            encoder_outputs=turn_audio_features,
                                            attention_mask=turn_audio_masks,
                                            text_encoder_only=True)
            elif self.cfg.use_video_mode:
                cur_text_features = self.model(inputs_embeds=cur_text_features,
                                            decoder_attention_mask=cur_text_masks,
                                            encoder_outputs=turn_video_features,
                                            attention_mask=turn_video_masks,
                                            text_encoder_only=True)
            for i in range(len(cur_index)):
                batch_text_node_features[cur_index[i]].append(cur_text_features[i][text_eos_indx[i]])

        for i in range(batch_size):
            batch_text_node_features[i] = torch.stack(batch_text_node_features[i])

            if not self.cfg.no_use_relation and len(batch[i]['text_history']) > 2:
                edge_index1=torch.LongTensor(batch_adj_matrix[i]).transpose(0,1)
                edge_index1=edge_index1.cuda(device=self.cfg.device)
                row,col=edge_index1.cpu()
                cc=torch.ones(row.shape[0]).cpu()
                d=1-(sp.coo_matrix((cc, (row, col)), shape=(len(batch[i]['text_history']), len(batch[i]['text_history'])))).toarray()
                d[d<1]=0
                c = torch.from_numpy(d).cuda()
                text_edges = batch_adj_matrix[i] + self.model.text_edge_relation(batch_text_node_features[i], c)
            else:
                text_edges = batch_adj_matrix[i]

            batch_text_edges.append(torch.LongTensor(text_edges).transpose(0,1).to(self.cfg.device))

        return batch_text_node_features, batch_text_edges, batch_text_features, batch_dia_emotion_forecast, batch_dia_emotion_forecast_edges, batch_audio_features, batch_video_features, batch_text_masks, batch_audio_edges, batch_video_edges, labels_ids, class_labels, sr_nos, sentiment_token_indexs, emotion_token_indexs, batch_input_ids
    
    def get_adjacency_matrix(self, node_num, edges):
        adjacency_matrix = torch.zeros((node_num, node_num))
        for i in range(len(edges[0])):
            a = edges[0][i]
            b = edges[1][i]
            adjacency_matrix[a][b] = 1
        adjacency_matrix = adjacency_matrix.to(self.cfg.device)
        return adjacency_matrix

    def similarity(self, x, y, t=1.0):
        return F.cosine_similarity(x, y, dim=-1) / t

    def contrastive_utterance_loss(self, batch_audio_features, batch_video_features):
        batch_size = len(batch_audio_features)
        loss = 0
        for i in range(batch_size):
            cur_audio = batch_audio_features[i][:-1]
            cur_video = batch_video_features[i][1:]
            cur_features = torch.cat([cur_audio, cur_video], dim=0)
            windows_size = len(cur_audio)
            for j in range(2 * windows_size):
                distances = 0
                for k in range(2 * windows_size):
                    if k != j:
                        distances += torch.exp(self.similarity(cur_features[j], cur_features[k]))
                g = (j + windows_size) if j < windows_size else (j - windows_size)
                loss += -torch.log(torch.exp(self.similarity(cur_features[j], cur_features[g])) / distances)
        return loss / batch_size
    
    def p_contrastive_utterance_loss(self, A, B, is_emotion=True):
        batch_size = len(A)
        loss = 0

        for i in range(batch_size):
            if is_emotion:
                cur_audio = A[i][:-1]
                cur_video = B[i][:-1]
            else:
                cur_audio = A[i]
                cur_video = B[i]
            windows_size = len(cur_audio)
            if windows_size < 1:
                continue
            
            cur_features = torch.cat([cur_audio, cur_video], dim=0)
            
            sim_matrix = torch.exp(self.similarity(cur_features.unsqueeze(1), cur_features.unsqueeze(0)))
            
            mask = torch.eye(2 * windows_size, device=cur_features.device).bool()
            sim_matrix = sim_matrix.masked_fill(mask, 0)

            distances = sim_matrix.sum(dim=1)
            
            positive_indices = torch.cat(
                [torch.arange(windows_size, 2 * windows_size, device=cur_features.device),
                torch.arange(0, windows_size, device=cur_features.device)]
            )
            
            positive_sim = sim_matrix[torch.arange(2 * windows_size, device=cur_features.device), positive_indices]

            loss += -torch.log(positive_sim / distances).sum()
        
        return loss / batch_size


    def train(self):
        num_training_steps_per_epoch = self.reader.train_steps
        
        if self.cfg.dataset == 'meld':
            optimizer, scheduler = self.get_optimizer_and_scheduler(
                        num_training_steps_per_epoch, self.cfg.s_learning_rate, self.cfg.epochs - 10, self.cfg.warmup_ratio, True)
        elif self.cfg.dataset == 'iemocap':
            optimizer, scheduler = self.get_optimizer_and_scheduler(
                        num_training_steps_per_epoch, self.cfg.s_learning_rate, self.cfg.epochs - 10, self.cfg.warmup_ratio, True)
        elif self.cfg.dataset == 'mustard':
            optimizer, scheduler = self.get_optimizer_and_scheduler(
                        num_training_steps_per_epoch, self.cfg.s_learning_rate, self.cfg.epochs - 10, self.cfg.warmup_ratio, True if (self.cfg.use_audio_mode or self.cfg.use_video_mode) else False)
        elif self.cfg.dataset == 'memotion':
            optimizer, scheduler = self.get_optimizer_and_scheduler(
                        num_training_steps_per_epoch, self.cfg.s_learning_rate, self.cfg.epochs - 10, self.cfg.warmup_ratio, True if (self.cfg.use_audio_mode or self.cfg.use_video_mode) else False)
        elif self.cfg.dataset == 'mosei':
            optimizer, scheduler = self.get_optimizer_and_scheduler(
                        num_training_steps_per_epoch, self.cfg.s_learning_rate, self.cfg.epochs - 10, self.cfg.warmup_ratio, True if (self.cfg.use_audio_mode or self.cfg.use_video_mode) else False)

        reporter = Reporter(self.cfg.log_frequency, self.logger)

        for epoch in (range(1, self.cfg.epochs + 1)):

            if (epoch - 1) == self.cfg.epochs - 10:
                optimizer, scheduler = self.get_optimizer_and_scheduler(
                    num_training_steps_per_epoch, self.cfg.learning_rate, 10, self.cfg.warmup_ratio, False)

            self.model.train()
            self.model.zero_grad()
            torch.set_grad_enabled(True)
            step = 0

            for batch in self.reader.get_data_iterator(self.reader.data['train'], shuffle=True, batch_size=self.cfg.batch_size):
                start_time = time.time()

                is_start = batch[1]
                if is_start:
                    history = []
                    emotion_history = []
                    for i in range(len(batch[0])):
                        history.append([])
                        emotion_history.append([])

                batch_text_node_features, batch_text_edges, batch_text_features, batch_dia_emotion_forecast, batch_dia_emotion_forecast_edges, batch_audio_features, batch_video_features, batch_text_masks, batch_audio_edges, batch_video_edges, labels_ids, class_labels, _, sentiment_token_indexs, emotion_token_indexs, c_ = self.process_batch(batch, None, None)

                con_u_loss = 0
                if not self.cfg.no_emotion_constract:
                    con_u_loss = self.p_contrastive_utterance_loss(batch_text_node_features, batch_dia_emotion_forecast)
                    if self.cfg.use_audio_mode:
                        con_u_loss += self.p_contrastive_utterance_loss(batch_audio_features, batch_dia_emotion_forecast)
                    if self.cfg.use_video_mode:
                        con_u_loss += self.p_contrastive_utterance_loss(batch_video_features, batch_dia_emotion_forecast)

                for cp in range(len(batch_dia_emotion_forecast)):
                    batch_dia_emotion_forecast[cp] = self.model.emotion_gat((batch_dia_emotion_forecast[cp], batch_dia_emotion_forecast_edges[cp]))[0]
                
                if self.cfg.use_gat:
                    for cp in range(len(batch_text_node_features)):
                        batch_text_node_features[cp] = self.model.text_gat((batch_text_node_features[cp], batch_text_edges[cp]))[0]

                if self.cfg.use_gat and self.cfg.use_audio_mode:
                    for cp in range(len(batch_audio_features)):
                        batch_audio_features[cp] = self.model.audio_gat((batch_audio_features[cp], batch_audio_edges[cp]))[0]
                
                if self.cfg.use_gat and self.cfg.use_video_mode:
                    for cp in range(len(batch_video_features)):
                        batch_video_features[cp] = self.model.video_gat((batch_video_features[cp], batch_video_edges[cp]))[0]
                
                con_c_loss = 0
                if not self.cfg.no_semantic_constract:
                    if self.cfg.use_audio_mode:
                        con_c_loss += self.p_contrastive_utterance_loss(batch_audio_features, batch_text_node_features, False)
                    if self.cfg.use_video_mode:
                        con_c_loss += self.p_contrastive_utterance_loss(batch_video_features, batch_text_node_features, False)
                    if self.cfg.use_audio_mode and self.cfg.use_video_mode:
                        con_c_loss += self.p_contrastive_utterance_loss(batch_audio_features, batch_video_features, False)
                
                batch_multi_features = []
                if self.cfg.use_audio_mode and self.cfg.use_video_mode:
                    for cp in range(len(batch_video_features)):
                        batch_multi_features.append(
                            torch.cat([batch_text_node_features[cp], batch_audio_features[cp], batch_video_features[cp]], dim=0)
                        )
                    batch_multi_features, batch_multi_masks, _ = self.pad_features(batch_multi_features, 'text')
                elif self.cfg.use_audio_mode:
                    for cp in range(len(batch_audio_features)):
                        batch_multi_features.append(
                            torch.cat([batch_text_node_features[cp], batch_audio_features[cp]], dim=0)
                        )
                    batch_multi_features, batch_multi_masks, _ = self.pad_features(batch_audio_features, 'text')
                elif self.cfg.use_video_mode:
                    for cp in range(len(batch_video_features)):
                        batch_multi_features.append(
                            torch.cat([batch_text_node_features[cp], batch_video_features[cp]], dim=0)
                        )
                    batch_multi_features, batch_multi_masks, _ = self.pad_features(batch_video_features, 'text')
                else:
                    batch_multi_features, batch_multi_masks, _ = self.pad_features(batch_text_node_features, 'text')
                    

                labels_ids = torch.stack(self.pad_batch_sequences(labels_ids))
                labels_ids = labels_ids.to(self.cfg.device)

                outputs = self.model(encoder_outputs=batch_text_features,
                                        attention_mask=batch_text_masks,
                                        audio_hidden_states=batch_multi_features,
                                        audio_attention_mask=batch_multi_masks,
                                        labels=labels_ids,
                                        return_dict=True)
                
                response_output = torch.argmax(outputs.logits, dim=-1).cpu().numpy().tolist()

                for i in range(len(batch[0])):
                    cur_label = labels_ids[i].cpu().numpy().tolist()
                    eos_idx = cur_label.index(self.reader.eos_token_id)
                    emotion_pred = self.reader.tokenizer.decode(response_output[i][emotion_token_indexs[i]])
                    sentiment_pred = self.reader.tokenizer.decode(response_output[i][sentiment_token_indexs[i]])

                    history[i].append(batch[0][i]['text_history'][-1][:-1] + response_output[i][11:eos_idx] + [self.reader.eos_token_id])
                    emotion_history[i].append(self.reader.encode_text(batch[0][i]['speaker'] + ": " + emotion_pred + ' & ' + sentiment_pred, self.reader.bos_token, self.reader.eos_token))

                loss = outputs[0]
                loss += self.cfg.con_w * (con_u_loss + con_c_loss)
                pred = torch.argmax(outputs[1], dim=-1)

                num_resp_correct, num_resp_count = self.count_tokens(
                    pred, labels_ids, pad_id=self.reader.pad_token_id)
                
                step_outputs = {"loss": loss.item(),
                                "correct": num_resp_correct.item(),
                                "count": num_resp_count.item()}

                if self.cfg.grad_accum_steps > 1:
                    loss = loss / self.cfg.grad_accum_steps

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.max_grad_norm)

                if (step + 1) % self.cfg.grad_accum_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    lr = scheduler.get_last_lr()[0]

                    if reporter is not None:
                        reporter.step(start_time, lr, step_outputs)
                    
                step += 1
            self.logger.info("done {}/{} epoch".format(epoch, self.cfg.epochs))
            cur_model_path = self.save_model(epoch)
            if not self.cfg.no_validation:
                dev_loss = self.validation()

    def validation(self):
        self.model.eval()
        reporter = Reporter(1000000, self.logger)
        torch.set_grad_enabled(False)

        all_loss = 0.0

        for batch in self.reader.get_data_iterator(self.reader.data['dev'], batch_size=self.cfg.batch_size):
            start_time = time.time()
            
            is_start = batch[1]
            if is_start:
                history = []
                emotion_history = []
                for i in range(len(batch[0])):
                    history.append([])
                    emotion_history.append([])

            batch_text_node_features, batch_text_edges, batch_text_features, batch_dia_emotion_forecast, batch_dia_emotion_forecast_edges, batch_audio_features, batch_video_features, batch_text_masks, batch_audio_edges, batch_video_edges, labels_ids, class_labels, _, sentiment_token_indexs, emotion_token_indexs, c_ = self.process_batch(batch, None, None)

            if self.cfg.use_gat:
                for cp in range(len(batch_text_node_features)):
                    batch_text_node_features[cp] = self.model.text_gat((batch_text_node_features[cp], batch_text_edges[cp]))[0]

            if self.cfg.use_gat and self.cfg.use_audio_mode:
                for cp in range(len(batch_audio_features)):
                    batch_audio_features[cp] = self.model.audio_gat((batch_audio_features[cp], batch_audio_edges[cp]))[0]
            
            if self.cfg.use_gat and self.cfg.use_video_mode:
                for cp in range(len(batch_video_features)):
                    batch_video_features[cp] = self.model.video_gat((batch_video_features[cp], batch_video_edges[cp]))[0]
            
            batch_multi_features = []
            if self.cfg.use_audio_mode and self.cfg.use_video_mode:
                for cp in range(len(batch_video_features)):
                    batch_multi_features.append(
                        torch.cat([batch_text_node_features[cp], batch_audio_features[cp], batch_video_features[cp]], dim=0)
                    )
                batch_multi_features, batch_multi_masks, _ = self.pad_features(batch_multi_features, 'text')
            elif self.cfg.use_audio_mode:
                for cp in range(len(batch_audio_features)):
                    batch_multi_features.append(
                        torch.cat([batch_text_node_features[cp], batch_audio_features[cp]], dim=0)
                    )
                batch_multi_features, batch_multi_masks, _ = self.pad_features(batch_audio_features, 'text')
            elif self.cfg.use_video_mode:
                for cp in range(len(batch_video_features)):
                    batch_multi_features.append(
                        torch.cat([batch_text_node_features[cp], batch_video_features[cp]], dim=0)
                    )
                batch_multi_features, batch_multi_masks, _ = self.pad_features(batch_video_features, 'text')
            else:
                batch_multi_features, batch_multi_masks, _ = self.pad_features(batch_text_node_features, 'text')

            labels_ids = torch.stack(self.pad_batch_sequences(labels_ids))
            labels_ids = labels_ids.to(self.cfg.device)

            outputs = self.model(encoder_outputs=batch_text_features,
                                    attention_mask=batch_text_masks,
                                    audio_hidden_states=batch_multi_features,
                                    audio_attention_mask=batch_multi_masks,
                                    labels=labels_ids,
                                    return_dict=True)
            
            response_output = torch.argmax(outputs.logits, dim=-1).cpu().numpy().tolist()

            for i in range(len(batch[0])):
                cur_label = labels_ids[i].cpu().numpy().tolist()
                eos_idx = cur_label.index(self.reader.eos_token_id)
                emotion_pred = self.reader.tokenizer.decode(response_output[i][emotion_token_indexs[i]])
                sentiment_pred = self.reader.tokenizer.decode(response_output[i][sentiment_token_indexs[i]])

                history[i].append(batch[0][i]['text_history'][-1][:-1] + response_output[i][11:eos_idx] + [self.reader.eos_token_id])
                emotion_history[i].append(self.reader.encode_text(batch[0][i]['speaker'] + ": " + emotion_pred + ' & ' + sentiment_pred, self.reader.bos_token, self.reader.eos_token))

            loss = outputs[0]
            pred = torch.argmax(outputs[1], dim=-1)
            num_resp_correct, num_resp_count = self.count_tokens(
                    pred, labels_ids, pad_id=self.reader.pad_token_id)
        
            all_loss += loss.item()
                
            step_outputs = {"loss": loss.item(),
                            "correct": num_resp_correct.item(),
                            "count": num_resp_count.item()}

            reporter.step(start_time, lr=None, step_outputs=step_outputs, is_train=False)

        reporter.info_stats("dev")

        return all_loss

    def predict(self):
        print(self.cfg)
        self.model.eval()

        early_stopping = True if self.cfg.beam_size > 1 else False

        predict_res = {}
        emotion_correct_count = 0
        sentiment_correct_count = 0

        for batch in self.reader.get_data_iterator(self.reader.data['test'], batch_size=self.cfg.test_batch_size):
            is_start = batch[1]
            if is_start:
                history = []
                emotion_history = []
                for i in range(len(batch[0])):
                    history.append([])
                    emotion_history.append([])
            batch_text_node_features, batch_text_edges, batch_text_features, batch_dia_emotion_forecast, batch_dia_emotion_forecast_edges, batch_audio_features, batch_video_features, batch_text_masks, batch_audio_edges, batch_video_edges, labels_ids, class_labels, sr_nos, sentiment_token_indexs, emotion_token_indexs, input_ids = self.process_batch(batch, None, None)
            

            with torch.no_grad():

                if self.cfg.use_gat:
                    for cp in range(len(batch_text_node_features)):
                        batch_text_node_features[cp] = self.model.text_gat((batch_text_node_features[cp], batch_text_edges[cp]))[0]

                if self.cfg.use_gat and self.cfg.use_audio_mode:
                    for cp in range(len(batch_audio_features)):
                        batch_audio_features[cp] = self.model.audio_gat((batch_audio_features[cp], batch_audio_edges[cp]))[0]
                
                if self.cfg.use_gat and self.cfg.use_video_mode:
                    for cp in range(len(batch_video_features)):
                        batch_video_features[cp] = self.model.video_gat((batch_video_features[cp], batch_video_edges[cp]))[0]
                
                batch_multi_features = []
                if self.cfg.use_audio_mode and self.cfg.use_video_mode:
                    for cp in range(len(batch_video_features)):
                        batch_multi_features.append(
                            torch.cat([batch_text_node_features[cp], batch_audio_features[cp], batch_video_features[cp]], dim=0)
                        )
                    batch_multi_features, batch_multi_masks, _ = self.pad_features(batch_multi_features, 'text')
                elif self.cfg.use_audio_mode:
                    for cp in range(len(batch_audio_features)):
                        batch_multi_features.append(
                            torch.cat([batch_text_node_features[cp], batch_audio_features[cp]], dim=0)
                        )
                    batch_multi_features, batch_multi_masks, _ = self.pad_features(batch_audio_features, 'text')
                elif self.cfg.use_video_mode:
                    for cp in range(len(batch_video_features)):
                        batch_multi_features.append(
                            torch.cat([batch_text_node_features[cp], batch_video_features[cp]], dim=0)
                        )
                    batch_multi_features, batch_multi_masks, _ = self.pad_features(batch_video_features, 'text')
                else:
                    batch_multi_features, batch_multi_masks, _ = self.pad_features(batch_text_node_features, 'text')

                labels_ids = torch.stack(self.pad_batch_sequences(labels_ids))
                labels_ids = labels_ids.to(self.cfg.device)

                outputs = self.model.generate(encoder_outputs=batch_text_features,
                                    attention_mask=batch_text_masks,
                                    audio_hidden_states=batch_multi_features,
                                    audio_attention_mask=batch_multi_masks,
                                    max_length=200,
                                    do_sample=self.cfg.do_sample,
                                    num_beams=self.cfg.beam_size,
                                    early_stopping=early_stopping,
                                    temperature=self.cfg.temperature,
                                    top_k=self.cfg.top_k,
                                    top_p=self.cfg.top_p,
                                    return_dict_in_generate=True)

                outputs = outputs.sequences.cpu().numpy().tolist()

                batch_decoded = []
                for output in outputs:
                    output = output[1:]
                    if self.reader.eos_token_id in output:
                        eos_idx = output.index(self.reader.eos_token_id)
                        output = output[:eos_idx]
                    batch_decoded.append(output)
                
                for i in range(len(batch[0])):
                    predict_res[sr_nos[i]] = {}
                    predict_res[sr_nos[i]]['input_ids'] = self.reader.tokenizer.decode(input_ids[i])
                    predict_res[sr_nos[i]]['resp'] = self.reader.tokenizer.decode(labels_ids[i])
                    predict_res[sr_nos[i]]['resp_gen'] = self.reader.tokenizer.decode(batch_decoded[i])

                    emotion_label = self.reader.tokenizer.decode(labels_ids[i][emotion_token_indexs[i]])
                    sentiment_label = self.reader.tokenizer.decode(labels_ids[i][sentiment_token_indexs[i]])

                    predict_res[sr_nos[i]]['emotion_label'] = emotion_label
                    predict_res[sr_nos[i]]['sentiment_label'] = sentiment_label

                    emotion_pred = None
                    sentiment_pred = None
                    if self.reader.bos_emotion in predict_res[sr_nos[i]]['resp_gen']:
                        words = predict_res[sr_nos[i]]['resp_gen'].split(' ')
                        emotion_pred = words[words.index(self.reader.bos_emotion) + 1]
                    if self.reader.bos_sentiment in predict_res[sr_nos[i]]['resp_gen']:
                        words = predict_res[sr_nos[i]]['resp_gen'].split(' ')
                        sentiment_pred = words[words.index(self.reader.bos_sentiment) + 1]

                    if emotion_pred is not None and sentiment_pred is not None:
                        predict_res[sr_nos[i]]['emotion_pred'] = emotion_pred
                        predict_res[sr_nos[i]]['sentiment_pred'] = sentiment_pred

                    history[i].append(batch[0][i]['text_history'][-1][:-1] + batch_decoded[i][11:] + [self.reader.eos_token_id])
                    emotion_history[i].append(self.reader.encode_text(batch[0][i]['speaker'] + ": " + emotion_pred + ' & ' + sentiment_pred, self.reader.bos_token, self.reader.eos_token))
        
        if self.cfg.output:
            save_json(predict_res, os.path.join(self.cfg.ckpt, self.cfg.output))
        try:
            scores = evaluate_metrics(os.path.join(self.cfg.ckpt, self.cfg.output), self.cfg.dataset, self.logger)
        except Exception:
            print('exception')

        self.logger.info(scores)
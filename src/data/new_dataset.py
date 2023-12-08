import torch
import numpy as np
import json
import csv
import os
import json
from itertools import chain
import torch.utils.data as data
import pickle as pkl
from transformers import AutoTokenizer
from sklearn.decomposition import PCA

def get_img_region_box(img_region_dir,id):
    region_feat = np.load(
        os.path.join(img_region_dir + '/_att',
                     id[:-4] + '.npz'))['feat']
    box = np.load(
        os.path.join(img_region_dir + '/_box', id[:-4] + '.npy'))

    return region_feat, box


def get_aesc_spans( dic):
    aesc_spans = []
    for x in dic:
        aesc_spans.append((x['from'], x['to'], x['polarity']))
    return aesc_spans


def get_gt_aspect_senti( dic):
    gt = []
    for x in dic:
        gt.append((' '.join(x['term']), x['polarity']))
    return gt

def getData(infos,split):
    infos = json.load(open(infos, 'r'))

    if split == 'train':
        data_set = json.load(
            open(infos['data_dir'] + '/train.json', 'r'))
        img_region_dir = infos['img_region_dir']
    elif split == 'dev':
        data_set = json.load(
            open(infos['data_dir'] + '/dev.json', 'r'))
        img_region_dir = infos['img_region_dir']
    elif split == 'test':
        data_set = json.load(
            open(infos['data_dir'] + '/test.json', 'r'))
        img_region_dir = infos['img_region_dir']
    else:
        raise RuntimeError("split type is not exist!!!")

    total_batch = []
    for data in data_set:
        output = {}
        img_id = data['image_id']
        region_feat, box = get_img_region_box(img_region_dir,img_id)
        # img_feature = np.concatenate([region_feat, box], axis=1)  #check 维度
        img_feature = region_feat
        u_pos = data['u_pos']
        output['u_pos'] = u_pos
        output['img_feat'] = img_feature

        output['sentence'] = ' '.join(data['words'])

        aesc_spans = get_aesc_spans(data['aspects'])
        output['aesc_spans'] = aesc_spans
        output['image_id'] = img_id
        gt = get_gt_aspect_senti(data['aspects'])
        output['gt'] = gt
        total_batch.append(output)
    return total_batch

class Collator:
    """
    The collator for all types of dataset.
    Remember to add the corresponding collation code after adding a new type of task.
    """
    def __init__(self,
                 tokenizer,
                 is_mlm=False,
                 has_label=True,
                 mlm_enabled=False,
                 mrm_enabled=False,
                 senti_enabled=False,
                 ae_enabled=False,
                 oe_enabled=False,
                 ae_oe_enabled=False,
                 aesc_enabled=False,
                 anp_enabled=False,
                 anp_generate_enabled=False,
                 twitter_ae_enabled=False,
                 twitter_sc_enabled=False,
                 text_only=False,
                 mlm_probability=0.0,
                 mrm_probability=0.0,
                 lm_max_len=30,
                 max_img_num=36,
                 max_span_len=20):
        """
        :param tokenizer: ConditionTokenizer
        :param mlm_enabled: bool, if use mlm for language modeling. False for autoregressive modeling
        :param mrm_enabled: bool, if use mrm
        :param rp_enabled: bool, if use relation prediction (VG)
        :param ap_enabled: bool, if use attribute prediction (VG)
        :param mlm_probability: float, probability to mask the tokens
        :param mrm_probability: float, probability to mask the regions
        """
        self._tokenizer = tokenizer
        self._has_label = has_label
        self._is_mlm = is_mlm
        self._mrm_enabled = mrm_enabled
        self._mlm_enabled = mlm_enabled
        self._senti_enabled = senti_enabled
        self._anp_enabled = anp_enabled
        self._anp_generate_enabled = anp_generate_enabled
        self._ae_enabled = ae_enabled
        self._oe_enabled = oe_enabled
        self._ae_oe_enabled = ae_oe_enabled
        self._aesc_enabled = aesc_enabled
        self._twitter_ae_enabled = twitter_ae_enabled
        self._twitter_sc_enabled = twitter_sc_enabled
        self._lm_max_len = lm_max_len
        self._max_img_num = max_img_num
        self._mlm_probability = mlm_probability
        self._mrm_probability = mrm_probability
        self._max_span_len = max_span_len
        self.text_only = text_only
        if mlm_enabled and not has_label:
            raise ValueError(
                'mlm_enabled can not be true while has_label is false. MLM need labels.'
            )

    def _clip_text(self, text, length):
        tokenized = []
        for i, word in enumerate(text.split()):
            if i == 0:
                bpes = self._tokenizer._base_tokenizer.tokenize(word)
            else:
                bpes = self._tokenizer._base_tokenizer.tokenize(
                    word, add_prefix_space=True)
            bpes = self._tokenizer._base_tokenizer.convert_tokens_to_ids(bpes)
            tokenized.append(bpes)
        _tokenized = list(chain(*tokenized))
        return self._tokenizer.get_base_tokenizer().decode(_tokenized[:length])

    def call(self, batch):
        batch = [entry for entry in batch if entry is not None]

        image_features = [
            torch.from_numpy(x['img_feat'][:self._max_img_num])
            if 'img_feat' in x else torch.empty(0) for x in batch
        ]

        img_num = [len(x) for x in image_features]
        u_pos = [x['u_pos'] for x in batch]
        target = [x['sentence'] for x in batch]
        sentence = list(target)

        encoded_conditions = self._tokenizer.encode_condition(
            img_num=img_num, sentence=sentence, text_only=self.text_only)

        input_ids = encoded_conditions['input_ids']
        output = {}
        if self._is_mlm:
            input_ids = self._mask_tokens(
                inputs=input_ids,
                input_mask=encoded_conditions['sentence_mask'])
        condition_img_mask = encoded_conditions['img_mask']

        if self._mrm_enabled:
            encode_mrm = self._tokenizer.encode_mrm([x['cls'] for x in batch])
            mrm_labels_all = encode_mrm['mrm_labels']
            probability_matrix = torch.full(input_ids.shape,
                                            self._mrm_probability,
                                            dtype=torch.float)
            masked_regions = torch.bernoulli(probability_matrix).bool()
            input_ids[masked_regions
                      & condition_img_mask] = self._tokenizer.cls_token_id
            decoder_input_ids = encode_mrm['mrm_decoder_input_ids']
            for i in range(input_ids.size(0)):
                for j in range(36):
                    if input_ids[i, j + 1] == self._tokenizer.cls_token_id:
                        decoder_input_ids[i, j +
                                          2] = self._tokenizer.cls_token_id
            mrm_labels = []
            for i in range(len(batch)):
                # create mrm_labels
                masked_indices = masked_regions[i][
                    condition_img_mask[i]].nonzero(as_tuple=False)
                mrm_label = mrm_labels_all[i]
                mrm_labels.append(mrm_label[masked_indices].clone())

                if len(image_features[i]) > 0:
                    image_features[i][masked_indices] = torch.zeros(
                        (len(masked_indices), 1, 2048),
                        dtype=image_features[i].dtype)
            MRM = {}
            MRM['mrm_labels'] = mrm_labels
            MRM['mrm_decoder_input_ids'] = decoder_input_ids
            MRM['mrm_masks'] = decoder_input_ids == self._tokenizer.cls_token_id
            MRM['mrm_decoder_attention_mask'] = encode_mrm[
                'mrm_decoder_attention_mask']
            output['MRM'] = MRM
            output['task'] = 'MRM'
        output['u_pos'] = u_pos
        output['input_ids'] = input_ids
        output['attention_mask'] = encoded_conditions['attention_mask']
        output['image_features'] = image_features
        output['input_ids'] = input_ids
        if self._has_label:
            # encode mrm and mlm labels
            if self._mlm_enabled:
                mlm_output = self._tokenizer.encode_label(label=target,
                                                          img_num=img_num)
                output['MLM'] = mlm_output
                output['task'] = 'MLM'

            if self._senti_enabled:
                output['Sentiment'] = self._tokenizer.encode_senti(
                    [x['sentiment'] for x in batch])
                output['task'] = 'Sentiment'

            if self._anp_generate_enabled:
                output['ANP_generate'] = self._tokenizer.encode_anp_generate(
                    [x['ANP_words'] for x in batch])
                output['task'] = 'ANP_generate'
            if self._aesc_enabled:
                output['AESC'] = self._tokenizer.encode_aesc(
                    target, [x['aesc_spans'] for x in batch],
                    self._max_span_len)
                output['task'] = 'AESC'
            if self._ae_oe_enabled:
                output['AE_OE'] = self._tokenizer.encode_ae_oe(
                    target, [x['aspect_spans'] for x in batch],
                    [x['opinion_spans'] for x in batch])
                output['task'] = 'AE_OE'

        output['image_id'] = [x['image_id'] for x in batch]
        output['gt'] = [x['gt'] for x in batch]
        return output

    def _mask_tokens(self, inputs, input_mask):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.

        :param inputs: torch.LongTensor, batch data
        :param input_mask: torch.Tensor, mask for the batch, False for the position with 0% probability to be masked
        """

        labels = inputs.clone()
        tokenizer = self._tokenizer.get_base_tokenizer()

        # We sample a few tokens in each sequence for masked-LM training
        probability_matrix = torch.full(labels.shape,
                                        self._mlm_probability,
                                        dtype=torch.float)
        special_tokens_mask = [
            tokenizer.get_special_tokens_mask(val,
                                              already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask,
                                                     dtype=torch.bool),
                                        value=0.0)
        if tokenizer.pad_token is not None:
            padding_mask = labels.eq(tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(
            labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced & input_mask] = tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(
            labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(tokenizer.vocab_size,
                                     labels.shape,
                                     dtype=torch.long)
        inputs[indices_random & input_mask] = random_words[indices_random
                                                           & input_mask]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs

class Preprocess:
    def __init__(self,args,output):
        self.args = args
        self.input_ids = output['input_ids']
        self.attention_mask = output['attention_mask']
        self.image_id = output['image_id']
        self.image_features = output['image_features']
        self.image_features = np.array(self.image_features,dtype=object)
        self.AESC = output['AESC']
        self.span_labels = self.AESC['labels']
        self.span_masks = self.AESC['masks']
        self.gt_spans = self.AESC['spans']
        self.gt_spans = np.array(self.gt_spans, dtype=object)
        self.task = output['task']
        self.u_pos = output['u_pos']
        # self.object_nums_path = args.object_nums_path
        # with open(self.object_nums_path, 'r', encoding='utf-8') as f:
        #     self.image_ids2object_num_dict = json.load(f)

        self.image_text_similarity_path = args.image_text_similarity_path
        self.image_text_region_similarity_path = args.image_text_region_similarity_path

        self.count_num = self.calculate_count_noun()
        self.sequence_id = np.arange(len(self.input_ids))
        # self.object_nums = self.calculate_object_num()

        self.image_text_similarity = self.calculate_image_text_similarity()
        self.image_text_region_similarity = self.calculate_image_text_region_similarity()

        # # 根据 object num 排序
        # self.input_ids_by_object_num, self.attention_masks_by_object_num, \
        # self.image_ids_by_object_num, self.image_feats_by_object_num, \
        # self.span_labels_by_object_num, self.span_masks_by_object_num, \
        # self.gt_spans_by_object_num = self.sort_by_object_num_difficulty()

        # # 根据 noun num 排序
        # self.input_ids_by_noun_num, self.attention_masks_by_noun_num, \
        # self.image_ids_by_noun_num, self.image_feats_by_noun_num, \
        # self.span_labels_by_noun_num, self.span_masks_by_noun_num, \
        # self.gt_spans_by_noun_num = self.sort_by_noun_num_difficulty()

        # 根据 similarity 排序
        self.input_ids_by_similarity, self.attention_masks_by_similarity, \
        self.image_ids_by_similarity, self.image_feats_by_similarity, \
        self.span_labels_by_similarity, self.span_masks_by_similarity, \
        self.gt_spans_by_similarity = self.sort_by_modal_similarity_difficulty()

        # 根据 region similarity 排序
        self.input_ids_by_region_similarity, self.attention_masks_by_region_similarity, \
        self.image_ids_by_region_similarity, self.image_feats_by_region_similarity, \
        self.span_labels_by_region_similarity, self.span_masks_by_region_similarity, \
        self.gt_spans_by_region_similarity = self.sort_by_modal_region_similarity_difficulty()

    def calculate_count_noun(self):
        count_num = []
        for upos in self.u_pos:
            temp_count = 0
            for pos in upos:
                if pos == 'NOUN' or pos == 'PRON' or pos == 'PROPN':
                    temp_count += 1
            count_num.append(temp_count)
        return count_num

    def calculate_object_num(self):
        object_nums = []
        image_ids_object_nums_dict = {}
        for input_id in self.image_id:
            image_id = input_id.split('.')[0]
            object_nums.append(self.image_ids2object_num_dict[image_id])

        return object_nums

    def calculate_image_text_similarity(self):
        with open(self.image_text_similarity_path,'r',encoding='utf-8')as f:
            similarity_dict = json.load(f)
        similarity_list = []
        for image_id in self.image_id:
            similarity_list.append(abs(similarity_dict[image_id]))
        return similarity_list

    def calculate_image_text_region_similarity(self):
        with open(self.image_text_region_similarity_path,'r',encoding='utf-8')as f:
            similarity_dict = json.load(f)
        similarity_list = []
        for image_id in self.image_id:
            similarity_list.append(abs(similarity_dict[image_id]))
        return similarity_list

    def sort_by_object_num_difficulty(self):

        sort_index = np.argsort(self.object_nums)
        image_ids = np.array(self.image_id,dtype=object)
        # 排序
        input_ids = self.input_ids[sort_index]
        atttention_masks = self.attention_mask[sort_index]
        image_ids = image_ids[sort_index]

        image_feats = self.image_features[sort_index]
        span_labels = self.span_labels[sort_index]
        span_masks = self.span_masks[sort_index]
        gt_spans = self.gt_spans[sort_index]

        return input_ids,atttention_masks,image_ids,image_feats,span_labels,span_masks,gt_spans

    def sort_by_noun_num_difficulty(self):

        sort_index = np.argsort(self.count_num)
        image_ids = np.array(self.image_id, dtype=object)
        # 排序
        input_ids = self.input_ids[sort_index]
        atttention_masks = self.attention_mask[sort_index]
        image_ids = image_ids[sort_index]
        image_feats = self.image_features[sort_index]
        span_labels = self.span_labels[sort_index]
        span_masks = self.span_masks[sort_index]
        gt_spans = self.gt_spans[sort_index]

        return input_ids, atttention_masks, image_ids, image_feats, span_labels, span_masks, gt_spans

    def sort_by_modal_similarity_difficulty(self):
        sort_index = np.argsort(self.image_text_similarity)


        image_ids = np.array(self.image_id, dtype=object)
        # 排序
        input_ids = self.input_ids[sort_index]
        atttention_masks = self.attention_mask[sort_index]
        image_ids = image_ids[sort_index]
        image_feats = self.image_features[sort_index]
        span_labels = self.span_labels[sort_index]
        span_masks = self.span_masks[sort_index]
        gt_spans = self.gt_spans[sort_index]

        return input_ids, atttention_masks, image_ids, image_feats, span_labels, span_masks, gt_spans

    def sort_by_modal_region_similarity_difficulty(self):
        sort_index = np.argsort(self.image_text_region_similarity)


        image_ids = np.array(self.image_id, dtype=object)
        # 排序
        input_ids = self.input_ids[sort_index]
        atttention_masks = self.attention_mask[sort_index]
        image_ids = image_ids[sort_index]
        image_feats = self.image_features[sort_index]
        span_labels = self.span_labels[sort_index]
        span_masks = self.span_masks[sort_index]
        gt_spans = self.gt_spans[sort_index]

        return input_ids, atttention_masks, image_ids, image_feats, span_labels, span_masks, gt_spans

    def get_sample_batch_by_object_num(self,lambda_init,current_epoch,total_epoch):
        if self.args.curriculum_pace == 'linear':
            current_index = int(
                (lambda_init + (1 - lambda_init) * current_epoch / total_epoch) * len(self.input_ids_by_similarity))
        elif self.args.curriculum_pace == 'square':
            current_index = int((lambda_init ** 2 + (1 - lambda_init ** 2) * current_epoch / total_epoch) * len(
                self.input_ids_by_similarity))

        current_index = min(len(self.input_ids_by_object_num)-1,current_index)
        batch = [self.input_ids_by_object_num[:current_index],
                 self.attention_masks_by_object_num[:current_index],
                 self.image_ids_by_object_num[:current_index],
                 self.image_feats_by_object_num[:current_index],
                 self.span_labels_by_object_num[:current_index],
                 self.span_masks_by_object_num[:current_index],
                 self.gt_spans_by_object_num[:current_index]]
        return batch

    def get_sample_batch_by_noun_num(self,lambda_init,current_epoch,total_epoch):
        if self.args.curriculum_pace == 'linear':
            current_index = int(
                (lambda_init + (1 - lambda_init) * current_epoch / total_epoch) * len(self.input_ids_by_similarity))
        elif self.args.curriculum_pace == 'square':
            current_index = int((lambda_init ** 2 + (1 - lambda_init ** 2) * current_epoch / total_epoch) * len(
                self.input_ids_by_similarity))

        current_index = min(len(self.input_ids_by_noun_num)-1,current_index)
        batch = [self.input_ids_by_noun_num[:current_index],
                 self.attention_masks_by_noun_num[:current_index],
                 self.image_ids_by_noun_num[:current_index],
                 self.image_feats_by_noun_num[:current_index],
                 self.span_labels_by_noun_num[:current_index],
                 self.span_masks_by_noun_num[:current_index],
                 self.gt_spans_by_noun_num[:current_index]]
        return batch

    def get_sample_batch_by_similarity(self,lambda_init,current_epoch,total_epoch):
        if self.args.curriculum_pace == 'linear':
            current_index = int((lambda_init + (1 - lambda_init) * current_epoch/total_epoch) * len(self.input_ids_by_similarity))
        elif self.args.curriculum_pace == 'square':
            current_index = int((lambda_init**2 + (1 - lambda_init**2) * current_epoch/total_epoch) * len(self.input_ids_by_similarity))
        current_index = min(len(self.input_ids_by_similarity)-1,current_index)
        batch = [self.input_ids_by_similarity[:current_index],
                 self.attention_masks_by_similarity[:current_index],
                 self.image_ids_by_similarity[:current_index],
                 self.image_feats_by_similarity[:current_index],
                 self.span_labels_by_similarity[:current_index],
                 self.span_masks_by_similarity[:current_index],
                 self.gt_spans_by_similarity[:current_index]]
        return batch

    def get_sample_batch_by_region_similarity(self,lambda_init,current_epoch,total_epoch):
        if self.args.curriculum_pace == 'linear':
            current_index = int((lambda_init + (1 - lambda_init) * current_epoch/total_epoch) * len(self.input_ids_by_similarity))
        elif self.args.curriculum_pace == 'square':
            current_index = int((lambda_init**2 + (1 - lambda_init**2) * current_epoch/total_epoch) * len(self.input_ids_by_similarity))
        current_index = min(len(self.input_ids_by_region_similarity)-1,current_index)
        batch = [self.input_ids_by_region_similarity[:current_index],
                 self.attention_masks_by_region_similarity[:current_index],
                 self.image_ids_by_region_similarity[:current_index],
                 self.image_feats_by_region_similarity[:current_index],
                 self.span_labels_by_region_similarity[:current_index],
                 self.span_masks_by_region_similarity[:current_index],
                 self.gt_spans_by_region_similarity[:current_index]]
        return batch

class Dataset(data.Dataset):
    def __init__(self,args,input_ids,attention_masks,image_feats,span_labels,span_masks):
        self.input_ids = torch.tensor(input_ids).to(args.device)
        self.attention_masks = torch.tensor(attention_masks).to(args.device)
        self.image_feats = image_feats
        self.span_labels = torch.tensor(span_labels).to(args.device)
        self.span_masks = torch.tensor(span_masks).to(args.device)
        self.data_ids = torch.arange(len(input_ids)).to(args.device)
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, index):
        return self.input_ids[index],self.attention_masks[index],self.image_feats[index],self.span_labels[index],\
               self.span_masks[index],self.data_ids[index]



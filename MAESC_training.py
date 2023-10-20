# -*- coding: UTF-8 -*-
import argparse
import json
import os
from datetime import datetime
from copy import deepcopy
from torch import optim
import torch
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW
import random
from src.data.collation import Collator

from src.data.new_dataset import *
from src.data.dataset import Twitter_Dataset
from src.data.tokenization_new import ConditionTokenizer
from src.model.config import MultiModalBartConfig
from src.model.MAESC_model import MultiModalBartModel_AESC
from src.model.model import MultiModalBartModelForPretrain
from src.training import fine_tune
from src.utils import Logger, save_training_data, load_training_data, setup_process, cleanup_process
from src.model.metrics import AESCSpanMetric
from src.model.generater import SequenceGeneratorModel
import src.eval_utils as eval_utils
import numpy as np
import torch.backends.cudnn as cudnn
from src.eval_utils import predict

def main(rank, args):
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    checkpoint_path = os.path.join(args.checkpoint_dir, timestamp)
    tb_writer = None
    add_name = ''
    log_dir = os.path.join(args.log_dir, timestamp + add_name)

    # make log dir and tensorboard writer if log_dir is specified
    if rank == 0 and args.log_dir is not None:
        os.makedirs(log_dir)
        tb_writer = SummaryWriter(log_dir=log_dir)

    logger = Logger(log_dir=os.path.join(log_dir, 'log.txt'),
                    enabled=(rank == 0))

    # make checkpoint dir if not exist
    if args.is_check == 1 and not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)
        logger.info('Made checkpoint directory: "{}"'.format(checkpoint_path))

    logger.info('Initialed with {} GPU(s)'.format(args.gpu_num), pad=True)
    for k, v in vars(args).items():
        logger.info('{}: {}'.format(k, v))

    # =========================== model =============================

    logger.info('Loading model...')

    if args.cpu:
        device = 'cpu'
        map_location = device
    else:
        device = torch.device("cuda:{}".format(rank))
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}

    tokenizer = ConditionTokenizer(args=args)
    label_ids = list(tokenizer.mapping2id.values())
    senti_ids = list(tokenizer.senti2id.values())

    if args.model_config is not None:
        bart_config = MultiModalBartConfig.from_dict(
            json.load(open(args.model_config)))
    else:
        bart_config = MultiModalBartConfig.from_pretrained(args.checkpoint)

    if args.dropout is not None:
        bart_config.dropout = args.dropout
    if args.attention_dropout is not None:
        bart_config.attention_dropout = args.attention_dropout
    if args.classif_dropout is not None:
        bart_config.classif_dropout = args.classif_dropout
    if args.activation_dropout is not None:
        bart_config.activation_dropout = args.activation_dropout

    bos_token_id = 0  # 因为是特殊符号
    eos_token_id = 1

    if args.checkpoint:
        pretrain_model = MultiModalBartModelForPretrain.from_pretrained(
            args.checkpoint,
            config=bart_config,
            bart_model=args.bart_model,
            tokenizer=tokenizer,
            label_ids=label_ids,
            senti_ids=senti_ids,
            args=args,
            error_on_mismatch=False)
        seq2seq_model = MultiModalBartModel_AESC(bart_config, args,
                                                 args.bart_model, tokenizer,
                                                 label_ids)
        seq2seq_model.encoder.load_state_dict(
            pretrain_model.encoder.state_dict())
        seq2seq_model.decoder.load_state_dict(
            pretrain_model.span_decoder.state_dict())
        model = SequenceGeneratorModel(seq2seq_model,
                                       bos_token_id=bos_token_id,
                                       eos_token_id=eos_token_id,
                                       max_length=args.max_len,
                                       max_len_a=args.max_len_a,
                                       num_beams=args.num_beams,
                                       do_sample=False,
                                       repetition_penalty=1,
                                       length_penalty=1.0,
                                       pad_token_id=eos_token_id,
                                       restricter=None)
    else:
        seq2seq_model = MultiModalBartModel_AESC(bart_config, args,
                                                 args.bart_model, tokenizer,
                                                 label_ids)
        model = SequenceGeneratorModel(seq2seq_model, bos_token_id=bos_token_id,
                                       eos_token_id=eos_token_id, max_length=args.max_len,
                                       max_len_a=args.max_len_a,num_beams=args.num_beams,do_sample=False, repetition_penalty=1,
                                       length_penalty=1.0,  pad_token_id=eos_token_id, restricter=None)

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scaler = GradScaler() if args.amp else None

    epoch = 0
    logger.info('Loading data...')
    collate_aesc = Collator(tokenizer,mlm_enabled=False, senti_enabled=False,ae_enabled=False,oe_enabled=False,aesc_enabled=True,
                            anp_enabled=False,text_only=args.text_only)

    train_total_data = getData(args.dataset[0][1], split='train')

    dev_total_data = getData(args.dataset[0][1], split='dev')
    test_total_data = getData(args.dataset[0][1], split='test')

    train_output = collate_aesc.call(train_total_data)
    dev_output = collate_aesc.call(dev_total_data)
    test_output = collate_aesc.call(test_total_data)

    train_p = Preprocess(args,train_output)

    dev_dataset = Dataset(args,dev_output['input_ids'],dev_output['attention_mask'],
                          dev_output['image_features'],dev_output['AESC']['labels'],dev_output['AESC']['masks'])

    dev_gt_spans = dev_output['AESC']['spans']
    dev_loader = DataLoader(dev_dataset,batch_size = args.batch_size)

    test_dataset = Dataset(args, test_output['input_ids'], test_output['attention_mask'],
                          test_output['image_features'],test_output['AESC']['labels'],test_output['AESC']['masks'])
    test_gt_spans = test_output['AESC']['spans']
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    callback = None
    metric = AESCSpanMetric(eos_token_id, num_labels=len(label_ids), conflict_id=-1)

    model.train()


    region_similarity_model = deepcopy(model)
    similarity_model = deepcopy(model)

    region_similarity_optimizer = AdamW(region_similarity_model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    similarity_optimizer = AdamW(similarity_model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    start = datetime.now()

    best_dev_region_similarity_res = None
    best_dev_region_similarity_model = None

    best_dev_similarity_res = None
    best_dev_similarity_model = None

    best_dev_res = None
    best_dev_model = None

    lambda_init = args.lambda_init # 初始比例，是一个重要的超参数
    region_similarity_epoch = 0
    similarity_epoch = 0

    # 当前的表现
    current_similarity_performance = 0
    current_region_similarity_performance = 0
    flag = True # 连续五轮表现不提高flag会变为false
    # 达到目标后是否继续训练
    total_step = args.total_step

    def train_by_region_similarity(current_step, total_step, model, optimizer):
        current_step = min(current_step,total_step)
        region_similarity_batch = train_p.get_sample_batch_by_region_similarity(lambda_init, current_step,
                                                                                total_step)
        input_ids, attention_masks, image_ids, image_feats, span_labels, span_masks, gt_spans = region_similarity_batch
        region_similarity_dataset = Dataset(args, input_ids, attention_masks, image_feats, span_labels, span_masks)
        region_similarity_dataloader = DataLoader(region_similarity_dataset, batch_size=args.batch_size, shuffle=True)

        fine_tune(epoch=epoch, model=model, train_loader=region_similarity_dataloader,
                  train_gt_span=gt_spans, metric=metric, optimizer=optimizer, args=args,
                  device=device, logger=logger, callback=callback, log_interval=1, tb_writer=tb_writer, tb_interval=1,
                  scaler=scaler)

    def train_by_similarity(current_step, total_step, model, optimizer):
        current_step = min(current_step, total_step)
        # 相似度
        similarity_batch = train_p.get_sample_batch_by_similarity(lambda_init, current_step, total_step)
        input_ids, attention_masks, image_ids, image_feats, span_labels, span_masks, gt_spans = similarity_batch
        similarity_dataset = Dataset(args, input_ids, attention_masks, image_feats, span_labels, span_masks)
        similarity_dataloader = DataLoader(similarity_dataset, batch_size=args.batch_size, shuffle=True)

        fine_tune(epoch=epoch, model=model, train_loader=similarity_dataloader,
                  train_gt_span=gt_spans, metric=metric, optimizer=optimizer,
                  args=args, device=device, logger=logger, callback=callback, log_interval=1,
                  tb_writer=tb_writer, tb_interval=1, scaler=scaler)

    print('Init...')
    region_step = 0
    similarity_step = 0
    train_by_region_similarity(region_step, total_step,region_similarity_model,region_similarity_optimizer)
    train_by_similarity(similarity_step, total_step,similarity_model,similarity_optimizer)
    # object num
    region_similarity_res_dev = eval_utils.eval(args, region_similarity_model, dev_loader, dev_gt_spans, metric, device)
    logger.info('Region Similarity DEV  aesc_p:{} aesc_r:{} aesc_f:{}'.format(
        region_similarity_res_dev['aesc_pre'], region_similarity_res_dev['aesc_rec'],
        region_similarity_res_dev['aesc_f']))
    diff_performace_region = region_similarity_res_dev['aesc_f'] - current_region_similarity_performance
    current_region_similarity_performance = region_similarity_res_dev['aesc_f']

    # similarity
    similarity_res_dev = eval_utils.eval(args, similarity_model, dev_loader, dev_gt_spans, metric, device)
    logger.info('Similarity DEV  aesc_p:{} aesc_r:{} aesc_f:{}'.format(
        similarity_res_dev['aesc_pre'], similarity_res_dev['aesc_rec'], similarity_res_dev['aesc_f']))
    diff_performace_similarity = similarity_res_dev['aesc_f'] - current_similarity_performance
    current_similarity_performance = similarity_res_dev['aesc_f']
    not_increase = 0
    while epoch < args.epochs or not_increase <= 5:

        logger.info('Epoch {}'.format(epoch + 1), pad=True)
        if diff_performace_region >= diff_performace_similarity:
            train_by_region_similarity(region_step, total_step, model, optimizer)
            res_dev = eval_utils.eval(args, model, dev_loader, dev_gt_spans, metric, device)
            logger.info('DEV  aesc_p:{} aesc_r:{} aesc_f:{}'.format(
                res_dev['aesc_pre'], res_dev['aesc_rec'], res_dev['aesc_f']))

            region_step += 1
            train_by_region_similarity(region_step, total_step, region_similarity_model, region_similarity_optimizer)
            region_similarity_res_dev = eval_utils.eval(args, region_similarity_model, dev_loader, dev_gt_spans, metric,
                                                    device)
            diff_performace_region = region_similarity_res_dev['aesc_f'] - current_region_similarity_performance
            current_region_similarity_performance = region_similarity_res_dev['aesc_f']

        else:
            train_by_similarity(similarity_step, total_step, model, optimizer)
            res_dev = eval_utils.eval(args, model, dev_loader, dev_gt_spans, metric, device)
            logger.info('DEV  aesc_p:{} aesc_r:{} aesc_f:{}'.format(
                res_dev['aesc_pre'], res_dev['aesc_rec'], res_dev['aesc_f']))

            similarity_step += 1
            train_by_similarity(similarity_step, total_step, similarity_model, similarity_optimizer)
            similarity_res_dev = eval_utils.eval(args, similarity_model, dev_loader, dev_gt_spans, metric, device)
            diff_performace_similarity = similarity_res_dev['aesc_f'] - current_similarity_performance
            current_similarity_performance = similarity_res_dev['aesc_f']

        if best_dev_res is None:
            best_dev_res = res_dev
            best_dev_model = model
            not_increase = 0
        else:
            if best_dev_res['aesc_f'] < res_dev['aesc_f']:
                best_dev_res = res_dev
                best_dev_model = model
                not_increase = 0
            else:
                not_increase += 1

        epoch += 1

    print('save best dev test model...')
    import pickle
    best_dev_test_res = eval_utils.eval(args, best_dev_model, test_loader, test_gt_spans, metric,
                                        device)

    tot_predict, total_labels, tot_span_gts = predict(args, best_dev_model, test_loader, test_gt_spans)

    with open(os.path.join(log_dir, 'tot_predict.pkl'), 'wb') as f:
        pickle.dump(tot_predict, f)
    with open(os.path.join(log_dir, 'total_labels.pkl'), 'wb') as f:
        pickle.dump(total_labels, f)
    with open(os.path.join(log_dir, 'tot_span_gts.pkl'), 'wb') as f:
        pickle.dump(tot_span_gts, f)

    logger.info("Training complete in: " + str(datetime.now() - start),
                pad=True)
    logger.info('---------------------------')



    logger.info('BEST DEV:-----')
    logger.info('BEST DEV  aesc_p:{} aesc_r:{} aesc_f:{}'.format(
        best_dev_res['aesc_pre'], best_dev_res['aesc_rec'],
        best_dev_res['aesc_f']))
    logger.info(best_dev_res)

    logger.info('BEST DEV TEST:-----')
    logger.info('BEST DEV--TEST  aesc_p:{} aesc_r:{} aesc_f:{}'.format(
        best_dev_test_res['aesc_pre'], best_dev_test_res['aesc_rec'],
        best_dev_test_res['aesc_f']))
    logger.info(best_dev_test_res)





def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        action='append',
                        nargs=2,
                        metavar=('DATASET_NAME', 'DATASET_PATH'),
                        required=True,
                        help='')
    # required

    parser.add_argument('--checkpoint_dir',
                        required=True,
                        type=str,
                        help='where to save the checkpoint')
    parser.add_argument('--lambda_init',
                        type=float,
                        default=0.4,
                        help='where to save the checkpoint')

    parser.add_argument('--image_text_similarity_path',
                        type=str,
                        default='/home/data_ti6_d/lich/CurriculumLearningMABSA/PreprocessData/similarity2015.json',
                        help='where to save the checkpoint')

    parser.add_argument('--image_text_region_similarity_path',
                        type=str,
                        default='/home/data_ti6_d/lich/CurriculumLearningMABSA/PreprocessData/similarity2015.json',
                        help='where to save the checkpoint')

    parser.add_argument('--object_nums_path',
                        type=str,
                        default= '/home/data_ti6_d/lich/CurriculumLearningMABSA/PreprocessData/object_nums2015.json',
                        help='where to save the checkpoint')
    parser.add_argument('--bart_model',
                        default='/home/data_ti6_d/lich/pretrain_model/bart-base',
                        type=str,
                        help='bart pretrain model')
    # path
    parser.add_argument(
        '--log_dir',
        default=None,
        type=str,
        help='path to output log files, not output to file if not specified')
    parser.add_argument('--model_config',
                        default=None,
                        type=str,
                        help='path to load model config')
    parser.add_argument('--device',
                        default='cuda',
                        type=str,)
    parser.add_argument('--text_only',
                        default=False,
                        type=bool,
                        help='if only input the text')
    parser.add_argument('--checkpoint',
                        default=None,
                        type=str,
                        help='name or path to load weights')
    parser.add_argument('--lr_decay_every',
                        default=4,
                        type=int,
                        help='lr_decay_every')
    parser.add_argument('--lr_decay_ratio',
                        default=0.8,
                        type=float,
                        help='lr_decay_ratio')
    # training and evaluation
    parser.add_argument('--epochs',
                        default=70,
                        type=int,
                        help='number of training epoch')
    parser.add_argument('--eval_every', default=1, type=int, help='eval_every')
    parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
    parser.add_argument('--num_beams',
                        default=4,
                        type=int,
                        help='level of beam search on validation')
    parser.add_argument(
        '--continue_training',
        action='store_true',
        help='continue training, load optimizer and epoch from checkpoint')
    parser.add_argument('--warmup', default=0.1, type=float, help='warmup')
    # dropout
    parser.add_argument(
        '--dropout',
        default=None,
        type=float,
        help=
        'dropout rate for the transformer. This overwrites the model config')
    parser.add_argument(
        '--classif_dropout',
        default=None,
        type=float,
        help=
        'dropout rate for the classification layers. This overwrites the model config'
    )
    parser.add_argument(
        '--attention_dropout',
        default=None,
        type=float,
        help=
        'dropout rate for the attention layers. This overwrites the model config'
    )
    parser.add_argument(
        '--activation_dropout',
        default=None,
        type=float,
        help=
        'dropout rate for the activation layers. This overwrites the model config'
    )

    # hardware and performance
    parser.add_argument('--grad_clip', default=5, type=float, help='grad_clip')
    parser.add_argument('--gpu_num',
                        default=1,
                        type=int,
                        help='number of GPUs in total')
    parser.add_argument('--cpu',
                        action='store_true',
                        help='if only use cpu to run the model')
    parser.add_argument('--amp',
                        action='store_true',
                        help='whether or not to use amp')
    parser.add_argument('--master_port',
                        type=str,
                        default='12355',
                        help='master port for DDP')
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='training batch size')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--num_workers',
                        type=int,
                        default=0,
                        help='#workers for data loader')
    parser.add_argument('--max_len', type=int, default=10, help='max_len')
    parser.add_argument('--max_len_a',
                        type=float,
                        default=0.6,
                        help='max_len_a')

    parser.add_argument('--bart_init',
                        type=int,
                        default=1,
                        help='use bart_init or not')

    parser.add_argument('--check_info',
                        type=str,
                        default='',
                        help='check path to save')
    parser.add_argument('--is_check',
                        type=int,
                        default=0,
                        help='save the model or not')
    parser.add_argument('--curriculum_pace',
                        type=str,
                        default='square',
                        choices=['square','linear'],
                        help='save the model or not')
    parser.add_argument('--total_step',
                       type=int,
                       default=35,)
    parser.add_argument('--task', type=str, default='', help='task type')
    args = parser.parse_args()

    if args.gpu_num != 1 and args.cpu:
        raise ValueError('--gpu_num are not allowed if --cpu is set to true')

    if args.checkpoint is None and args.model_config is None:
        raise ValueError(
            '--model_config and --checkpoint cannot be empty at the same time')

    return args


if __name__ == '__main__':
    args = parse_args()

    # mp.spawn(main, args=(args, ), nprocs=args.gpu_num, join=True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cudnn.deterministic = True
    main(0, args)

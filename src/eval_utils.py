import torch
import torch.nn as nn


def eval(args, model, loader, gt_spans,metric, device):
    model.eval()
    for i, batch in enumerate(loader):
        # Forward pass
        # if args.task == 'twitter_ae':
        #     aesc_infos = {
        #         key: value
        #         for key, value in batch['TWITTER_AE'].items()
        #     }
        # elif args.task == 'twitter_sc':
        #     aesc_infos = {
        #         key: value
        #         for key, value in batch['TWITTER_SC'].items()
        #     }
        # else:
        #     aesc_infos = {key: value for key, value in batch['AESC'].items()}
        input_ids,attention_masks,image_feats,span_labels, \
        span_masks, data_ids = batch
        batch_gt_spans = []
        for j_id in data_ids:
            batch_gt_spans.append(gt_spans[j_id])
        mner_encode = {}
        mner_encode['labels'] = span_labels
        mner_encode['masks'] = span_masks
        mner_encode['spans'] = batch_gt_spans
        predict = model.predict(
            input_ids=input_ids.to(device),
            image_features=list(
                map(lambda x: x.to(device), image_feats)),
            attention_mask=attention_masks.to(device),
            aesc_infos=mner_encode)

        metric.evaluate(batch_gt_spans, predict,
                        span_labels.to(device))
        # break

    res = metric.get_metric()
    model.train()
    return res

def predict(args, model, test_dataloader,gt_spans):
    # 给出预测的 mner index 值
    model.eval()
    device = args.device
    tot_predict = []
    tot_span_gts  = []
    total_labels = []
    for i, batch in enumerate(test_dataloader):
        input_ids, attention_masks, image_feats, span_labels, \
        span_masks, data_ids = batch
        batch_gt_spans = []
        for j_id in data_ids:
            batch_gt_spans.append(gt_spans[j_id])
        mner_encode = {}
        mner_encode['labels'] = span_labels
        mner_encode['masks'] = span_masks
        mner_encode['spans'] = batch_gt_spans
        predict = model.predict(
            input_ids=input_ids.to(device),
            image_features=list(
                map(lambda x: x.to(device), image_feats)),
            attention_mask=attention_masks.to(device),
            aesc_infos=mner_encode)
        tot_predict.append(predict.cpu())
        total_labels.append(mner_encode['labels'])
        tot_span_gts.append(mner_encode['spans'])

    return tot_predict,total_labels,tot_span_gts
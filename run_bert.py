import torch
import numpy as np
import warnings
import csv
from pathlib import Path
from argparse import ArgumentParser
from pybert.train.losses import BCEWithLogLoss
from pybert.train.trainer import Trainer
from torch.utils.data import DataLoader
from pybert.io.bert_processor import BertProcessor
from pybert.common.tools import init_logger, logger
from pybert.common.tools import seed_everything
from pybert.configs.basic_config import config
from pybert.model.nn.bert_for_multi_label import BertForMultiLable
from pybert.preprocessing.preprocessor import EnglishPreProcessor
from pybert.callback.modelcheckpoint import ModelCheckpoint
from pybert.callback.trainingmonitor import TrainingMonitor
from pybert.train.metrics import AUC, AccuracyThresh, MultiLabelReport,Recall,Acc
from pytorch_transformers import AdamW, WarmupLinearSchedule
from torch.utils.data import RandomSampler, SequentialSampler


warnings.filterwarnings("ignore")


def run_train(args):
    # --------- data
    processor = BertProcessor(vocab_path=config['bert_vocab_path'], do_lower_case=args.do_lower_case)
    label_list = processor.get_labels()
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}

    train_data = processor.get_train(config['data_dir'] / f"{args.data_name}.train.pkl")
    train_examples = processor.create_examples(lines=train_data,
                                               example_type='train',
                                               cached_examples_file=config[
                                                    'data_dir'] / f"cached_train_examples_{args.arch}")
    train_features = processor.create_features(examples=train_examples,
                                               max_seq_len=args.train_max_seq_len,
                                               cached_features_file=config[
                                                    'data_dir'] / "cached_train_features_{}_{}".format(
                                                   args.train_max_seq_len, args.arch
                                               ))
#    print(train_features[0].input_ids)
#    print(train_features[0].segment_ids)
#    print(train_features[0].label_id)
#    print(train_features[0].input_mask)
    train_dataset = processor.create_dataset(train_features, is_sorted=args.sorted)
    if args.sorted:
        train_sampler = SequentialSampler(train_dataset)
    else:
        train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    valid_data = processor.get_dev(config['data_dir'] / f"{args.data_name}.valid.pkl")
    valid_examples = processor.create_examples(lines=valid_data,
                                               example_type='valid',
                                               cached_examples_file=config[
                                                                        'data_dir'] / f"cached_valid_examples_{args.arch}")

    valid_features = processor.create_features(examples=valid_examples,
                                               max_seq_len=args.eval_max_seq_len,
                                               cached_features_file=config[
                                                                        'data_dir'] / "cached_valid_features_{}_{}".format(
                                                   args.eval_max_seq_len, args.arch
                                               ))
    valid_dataset = processor.create_dataset(valid_features)
    valid_sampler = SequentialSampler(valid_dataset)
    valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.eval_batch_size)

    # ------- model
    logger.info("initializing model")
    if args.resume_path:
        args.resume_path = Path(args.resume_path)
        model = BertForMultiLable.from_pretrained(args.resume_path)
        # model.unfreeze(0,11)
    else:
        model = BertForMultiLable.from_pretrained(config['bert_model_dir'])
        # model.unfreeze(0,11)
    t_total = int(len(train_dataloader) / args.gradient_accumulation_steps * args.epochs)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    warmup_steps = int(t_total * args.warmup_proportion)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    lr_scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # ---- callbacks
    logger.info("initializing callbacks")
    train_monitor = TrainingMonitor(file_dir=config['figure_dir'], arch=args.arch)
    model_checkpoint = ModelCheckpoint(checkpoint_dir=config['checkpoint_dir'],mode=args.mode,
                                       monitor=args.monitor,arch=args.arch,
                                       save_best_only=args.save_best)

    # **************************** training model ***********************
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    trainer = Trainer(n_gpu=args.n_gpu,
                      model=model,
                      epochs=args.epochs,
                      logger=logger,
                      criterion=BCEWithLogLoss(reduction='sum'),
                      optimizer=optimizer,
                      lr_scheduler=lr_scheduler,
                      early_stopping=None,
                      training_monitor=train_monitor,
                      fp16=args.fp16,
                      resume_path=args.resume_path,
                      grad_clip=args.grad_clip,
                      model_checkpoint=model_checkpoint,
                      gradient_accumulation_steps=args.gradient_accumulation_steps,
                      batch_metrics=[AccuracyThresh(thresh=0.5),Recall(),Acc()],
                      epoch_metrics=[#AUC(average='micro', task_type='binary'),
                                     #MultiLabelReport(id2label=id2label)
                                     AccuracyThresh(thresh=0.5),
                                     Recall(),Acc()])
    trainer.train(train_data=train_dataloader, valid_data=valid_dataloader, seed=args.seed)


def run_test(args,test=False,k=7,med_map='pybert/dataset/med_map.csv'):
    from pybert.io.task_data import TaskData
    from pybert.test.predictor import Predictor
    data = TaskData()
    targets, sentences = data.read_data (raw_data_path=config['test_path'],
                                        preprocessor=EnglishPreProcessor(),
                                        is_train=test)
    print(f'-----------------------------------------\ntargets {targets}\n---------------------------------------------------')
    lines = list(zip(sentences, targets))
    processor = BertProcessor(vocab_path=config['bert_vocab_path'], do_lower_case=args.do_lower_case)
    label_list = processor.get_labels()
    id2label = {i: label for i, label in enumerate(label_list)}

    test_data = processor.get_test(lines=lines)
    test_examples = processor.create_examples(lines=test_data,
                                              example_type='test',
                                              cached_examples_file=config[
                                                                       'data_dir'] / f"cached_test_examples_{args.arch}")
    test_features = processor.create_features(examples=test_examples,
                                              max_seq_len=args.eval_max_seq_len,
                                              cached_features_file=config[
                                                                       'data_dir'] / "cached_test_features_{}_{}".format(
                                                  args.eval_max_seq_len, args.arch
                                              ))
    test_dataset = processor.create_dataset(test_features)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.train_batch_size)
    model = BertForMultiLable.from_pretrained(config['checkpoint_dir'])

    # ----------- predicting
    logger.info('model predicting....')
    predictor = Predictor(model=model,
                          logger=logger,
                          n_gpu=args.n_gpu, test=test)
    if test:
        results,targets = predictor.predict(data=test_dataloader)
    #print(f'results {results.shape}')
    #print(f'targets {targets.shape}')
        result =  dict()
        metrics = [Recall(),Acc()]
        for metric in metrics:
                metric.reset()
                metric(logits=results, target=targets)
                value = metric.value()
                if value is not None:
                    result[f'valid_{metric.name()}'] = value
        return result            
    else:
        results = predictor.predict(data=test_dataloader)
        pred = np.argsort(results)[:,-k:][:,::-1]
        with open('pybert/dataset/med_map.csv', mode='r') as infile:
          reader = csv.reader(infile)
          med_dict = {int(rows[0]):rows[1] for rows in reader}
          pred = np.vectorize(med_dict.get)(pred)
          return pred


def main():
    parser = ArgumentParser()
    parser.add_argument("--arch", default='bert', type=str)
    parser.add_argument("--do_data", action='store_true')
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_test", action='store_true')
    parser.add_argument("--save_best", action='store_true')
    parser.add_argument("--do_lower_case", action='store_true')
    parser.add_argument('--data_name', default='kaggle', type=str)
    parser.add_argument("--epochs", default=6, type=int)
    parser.add_argument("--resume_path", default='', type=str)
    parser.add_argument("--mode", default='min', type=str)
    parser.add_argument("--monitor", default='valid_loss', type=str)
    parser.add_argument("--valid_size", default=0.2, type=float)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--sorted", default=1, type=int, help='1 : True  0:False ')
    parser.add_argument("--n_gpu", type=str, default='0', help='"0,1,.." or "0" or "" ')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument('--eval_batch_size', default=8, type=int)
    parser.add_argument("--train_max_seq_len", default=256, type=int)
    parser.add_argument("--eval_max_seq_len", default=256, type=int)
    parser.add_argument('--loss_scale', type=float, default=0)
    parser.add_argument("--warmup_proportion", default=0.1, type=int, )
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--grad_clip", default=1.0, type=float)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--fp16_opt_level', type=str, default='O1')

    args = parser.parse_args()
    config['checkpoint_dir'] = config['checkpoint_dir'] / args.arch
    config['checkpoint_dir'].mkdir(exist_ok=True)
    # Good practice: save your training arguments together with the trained model
    torch.save(args, config['checkpoint_dir'] / 'training_args.bin')
    seed_everything(args.seed)
    init_logger(log_file=config['log_dir'] / f"{args.arch}.log")

    logger.info("Training/evaluation parameters %s", args)

    if args.do_data:
        from pybert.io.task_data import TaskData
        data = TaskData()
        targets, sentences = data.read_data(raw_data_path=config['raw_data_path'],
                                            preprocessor=EnglishPreProcessor(),
                                            is_train=True)
        data.train_val_split(X=sentences, y=targets, shuffle=True, stratify=False,
                             valid_size=args.valid_size, data_dir=config['data_dir'],
                             data_name=args.data_name)
    if args.do_train:
        run_train(args)

    if args.do_test:
      test = False
      pred = run_test(args,test=test)
      if(test):
        print(f'{pred}\n\n')
      else:
        print("\n")
        for pat in pred:
          for i,med in enumerate(pat):
            print(f'{i+1} {med}\n')
          print("\n\n")


if __name__ == '__main__':
    main()
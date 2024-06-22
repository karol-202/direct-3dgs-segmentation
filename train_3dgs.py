# Partially from: https://github.com/yanx27/Pointnet_Pointnet2_pytorch

import argparse
import os
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import time

from data_utils.extra_feature import ExtraFeature
from datasets.composed_3dgs_dataset import Composed3DGSDataset
from data_utils.utils import read_split_file
from datasets.composed_mesh_dataset import ComposedMeshDataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
CLASSES = [
    'bathtub',
    'bed',
    'chair',
    'desk',
    'dresser',
    'monitor',
    'night_stand',
    'sofa',
    'table',
    'toilet'
]
NUM_CLASSES = len(CLASSES)
CLASS2LABEL = {cls: i for i, cls in enumerate(CLASSES)}

sys.path.append(os.path.join(ROOT_DIR, 'models'))

class2label = {cls: i for i, cls in enumerate(CLASSES)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('data_path', type=str, help='path to the data directory')
    parser.add_argument('--model', type=str, default='pointnet_sem_seg', help='model name [default: pointnet_sem_seg]')
    parser.add_argument('--dataset_type', type=str, default='3DGS', help='Dataset type: 3DGS or SampledMesh [default: 3DGS]')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch', default=32, type=int, help='Epoch to run [default: 32]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--weight_decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int, default=4096, help='Point Number [default: 4096]')
    parser.add_argument('--lr_step_size', type=int, default=10, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--eval_after_epoch', action='store_true', help='Whether to evaluate after every epoch [default: False]')
    parser.add_argument('--extra_features', type=str, default=None, help='Comma separated extra 3DGS features to use (only for 3DGS) [default: None]')
    parser.add_argument('--sampling', type=str, default='uniform', help='Sampling method: uniform or fps (only for 3DGS) [default: uniform]')

    return parser.parse_args()


class TrainEnv:
    def log_string(self, str):
        self.logger.info(str)
        print(str)


def create_environment(args, train_paths, test_paths):
    env = TrainEnv()

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('sem_seg')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    env.checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    env.checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    env.logger = logging.getLogger("Model")
    env.logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    env.logger.addHandler(file_handler)
    env.log_string('PARAMETER ...')
    env.log_string(args)

    env.writer = SummaryWriter(log_dir=str(experiment_dir.joinpath('tensorboard')))

    num_point = args.npoint
    batch_size = args.batch_size
    extra_features = [ExtraFeature.feature_by_name(feature_name) for feature_name in args.extra_features] if args.extra_features is not None else []
    sampling = args.sampling

    print("start loading training data ...")
    if args.dataset_type == '3DGS':
        env.train_dataset = Composed3DGSDataset(model_paths=train_paths, class2label=CLASS2LABEL, sampling=sampling,
                                                num_point=num_point, extra_features=extra_features)
    elif args.dataset_type == 'SampledMesh':
        env.train_dataset = ComposedMeshDataset(model_paths=train_paths, class2label=CLASS2LABEL,
                                                num_point=num_point)
    print("start loading test data ...")
    if args.dataset_type == '3DGS':
        env.test_dataset = Composed3DGSDataset(model_paths=test_paths, class2label=CLASS2LABEL, sampling=sampling,
                                               num_point=num_point, extra_features=extra_features)
    elif args.dataset_type == 'SampledMesh':
        env.test_dataset = ComposedMeshDataset(model_paths=test_paths, class2label=CLASS2LABEL,
                                                num_point=num_point)

    env.trainDataLoader = torch.utils.data.DataLoader(env.train_dataset, batch_size=batch_size, shuffle=True,
                                                  pin_memory=True, drop_last=True, num_workers=4,
                                                  worker_init_fn=lambda x: np.random.seed(x + int(time.time())))
    env.testDataLoader = torch.utils.data.DataLoader(env.test_dataset, batch_size=batch_size, shuffle=False,
                                                 pin_memory=True, drop_last=True, num_workers=4)

    env.weights = torch.Tensor(env.train_dataset.label_weights).cuda()

    env.log_string("The number of training data is: %d" % len(env.train_dataset))
    env.log_string("The number of test data is: %d" % len(env.test_dataset))

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)

    env.classifier = MODEL.get_model(NUM_CLASSES, env.train_dataset.get_channels_count).cuda()
    env.classifier.apply(inplace_relu)
    env.criterion = MODEL.get_loss().cuda()

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/model.pth')
        env.start_epoch = checkpoint['epoch'] + 1
        env.classifier.load_state_dict(checkpoint['model_state_dict'])
        env.log_string('Use pretrain model')
    except:
        env.log_string('No existing model, starting from scratch...')
        env.start_epoch = 0
        env.classifier = env.classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        env.optimizer = torch.optim.Adam(
            env.classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.weight_decay_rate
        )
    else:
        env.optimizer = torch.optim.SGD(env.classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    return env


def close_environment(env):
    env.writer.close()

    handlers = env.logger.handlers[:]
    for handler in handlers:
        env.logger.removeHandler(handler)
        handler.close()


def train(env, args):
    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.lr_step_size

    global_epoch = 0
    best_iou = 0

    for epoch in range(env.start_epoch, args.epoch):
        env.log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.lr_step_size)), LEARNING_RATE_CLIP)
        env.log_string('Learning rate:%f' % lr)
        for param_group in env.optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        env.classifier = env.classifier.apply(lambda x: bn_momentum_adjust(x, momentum))

        env.writer.add_scalar('LR', lr, epoch)

        num_batches = len(env.trainDataLoader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        env.classifier = env.classifier.train()

        for i, (points, target) in tqdm(enumerate(env.trainDataLoader), total=len(env.trainDataLoader), smoothing=0.9):
            env.optimizer.zero_grad()

            points = points.data.numpy()
            points = torch.Tensor(points)
            points, target = points.float().cuda(), target.long().cuda()
            points = points.transpose(2, 1)

            seg_pred, trans_feat = env.classifier(points)
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

            batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
            target = target.view(-1, 1)[:, 0]
            loss = env.criterion(seg_pred, target, trans_feat, env.weights)
            loss.backward()
            env.optimizer.step()

            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
            correct = np.sum(pred_choice == batch_label)
            total_correct += correct
            total_seen += (args.batch_size * args.npoint)
            loss_sum += loss

        training_loss = loss_sum / num_batches
        training_accuracy = total_correct / float(total_seen)

        env.log_string('Training mean loss: %f' % training_loss)
        env.log_string('Training accuracy: %f' % training_accuracy)

        env.writer.add_scalar('Train loss', training_loss, epoch)
        env.writer.add_scalar('Train accuracy', training_accuracy, epoch)

        if epoch % 5 == 0 or epoch == args.epoch - 1:
            env.log_string('Save model...')
            savepath = str(env.checkpoints_dir) + '/model.pth'
            env.log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'model_state_dict': env.classifier.state_dict(),
                'optimizer_state_dict': env.optimizer.state_dict(),
            }
            torch.save(state, savepath)
            env.log_string('Saving model....')

        if args.eval_after_epoch:
            env.log_string('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
            eval_results = evaluate(env, args)

            env.log_string('eval point avg class IoU: %f' % eval_results.mIoU)
            env.log_string('eval point avg class acc: %f' % (
                eval_results.mean_class_accuracy))

            iou_per_class_str = '------- IoU --------\n'
            for l in range(NUM_CLASSES):
                iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f \n' % (
                    seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])),
                    eval_results.labelweights[l - 1],
                    eval_results.class_mIoU[l]
                )

            env.log_string(iou_per_class_str)
            env.log_string('Eval mean loss: %f' % eval_results.loss)
            env.log_string('Eval accuracy: %f' % eval_results.accuracy)

            env.writer.add_scalar('Eval loss', eval_results.loss, epoch)
            env.writer.add_scalar('Eval accuracy', eval_results.accuracy, epoch)
            env.writer.add_scalar('Eval mIoU', eval_results.mIoU, epoch)

            if eval_results.mIoU >= best_iou:
                best_iou = eval_results.mIoU
                env.log_string('Save model...')
                savepath = str(env.checkpoints_dir) + '/best_model.pth'
                env.log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'class_avg_iou': eval_results.mIoU,
                    'model_state_dict': env.classifier.state_dict(),
                    'optimizer_state_dict': env.optimizer.state_dict(),
                }
                torch.save(state, savepath)
                env.log_string('Saving model....')
            env.log_string('Best mIoU: %f' % best_iou)

        env.writer.flush()
        global_epoch += 1


class EvalResults:
    def __init__(self, mIoU, loss, accuracy, labelweights, mean_class_accuracy, class_mIoU):
        self.mIoU = mIoU
        self.loss = loss
        self.accuracy = accuracy
        self.labelweights = labelweights
        self.mean_class_accuracy = mean_class_accuracy
        self.class_mIoU = class_mIoU


def evaluate(env, args):
    with torch.no_grad():
        num_batches = len(env.testDataLoader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        labelweights = np.zeros(NUM_CLASSES)
        total_seen_class = [0 for _ in range(NUM_CLASSES)]
        total_correct_class = [0 for _ in range(NUM_CLASSES)]
        total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
        env.classifier = env.classifier.eval()

        for i, (points, target) in tqdm(enumerate(env.testDataLoader), total=len(env.testDataLoader), smoothing=0.9):
            points = points.data.numpy()
            points = torch.Tensor(points)
            points, target = points.float().cuda(), target.long().cuda()
            points = points.transpose(2, 1)

            seg_pred, trans_feat = env.classifier(points)
            pred_val = seg_pred.contiguous().cpu().data.numpy()
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

            batch_label = target.cpu().data.numpy()
            target = target.view(-1, 1)[:, 0]
            loss = env.criterion(seg_pred, target, trans_feat, env.weights)
            loss_sum += loss
            pred_val = np.argmax(pred_val, 2)
            correct = np.sum((pred_val == batch_label))
            total_correct += correct
            total_seen += (args.batch_size * args.npoint)
            tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
            labelweights += tmp

            for l in range(NUM_CLASSES):
                total_seen_class[l] += np.sum((batch_label == l))
                total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))

        labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))

        mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=float) + 1e-6))
        eval_loss = loss_sum / float(num_batches)
        eval_accuracy = total_correct / float(total_seen)
        mean_class_accuracy = np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=float) + 1e-6))
        class_mIoU = [total_correct_class[l] / float(total_iou_deno_class[l]) for l in range(NUM_CLASSES)]
        return EvalResults(mIoU, eval_loss, eval_accuracy, labelweights, mean_class_accuracy, class_mIoU)


def main(args):
    data_path = args.data_path
    train_scene_paths = read_split_file(data_path, 'train.txt')
    test_scene_paths = read_split_file(data_path, 'test.txt')
    env = create_environment(args, train_scene_paths, test_scene_paths)
    train(env, args)
    close_environment(env)


if __name__ == '__main__':
    main(parse_args())

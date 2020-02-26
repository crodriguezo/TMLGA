import os
import data
import json
import torch
import solver
import modeling
import numpy as np


from utils.visualization import Visualization
from utils.miscellaneous import mkdir

from tensorboardX import SummaryWriter

def trainer(cfg):
    print('trainer')
    dataloader_train, dataset_size_train = data.make_dataloader(cfg, is_train=True)
    dataloader_test, dataset_size_test   = data.make_dataloader(cfg, is_train=False)

    model = modeling.build(cfg)
    model.cuda()
    #model = torch.load("/home/crodriguezo/projects/phd/moment-localization-with-NLP/mlnlp_lastversion/checkpoints/anet_config7/model_epoch_80")
    optimizer = solver.make_optimizer(cfg, model)

    vis_train = Visualization(cfg, dataset_size_train)
    vis_test  = Visualization(cfg, dataset_size_test, is_train=False)

    writer_path = os.path.join(cfg.VISUALIZATION_DIRECTORY, cfg.EXPERIMENT_NAME)
    writer = SummaryWriter(writer_path)

    total_iterations = 0
    total_iterations_val = 0

    for epoch in range(cfg.EPOCHS):
        print("Epoch {}".format(epoch))
        model.train()
        for iteration, batch in enumerate(dataloader_train):
            index     = batch[0]

            videoFeat = batch[1].cuda()
            videoFeat_lengths = batch[2].cuda()

            tokens         = batch[3].cuda()
            tokens_lengths = batch[4].cuda()

            start    = batch[5].cuda()
            end      = batch[6].cuda()

            localiz  = batch[7].cuda()
            localiz_lengths = batch[8]
            time_starts = batch[9]
            time_ends = batch[10]
            factors = batch[11]
            fps = batch[12]

            loss, individual_loss, pred_start, pred_end, attention, atten_loss = model(videoFeat, videoFeat_lengths, tokens, tokens_lengths, start, end, localiz)
            print("Loss :{}".format(loss))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            vis_train.run(index, pred_start, pred_end, start, end, videoFeat_lengths, epoch, loss.detach(), individual_loss, attention, atten_loss, time_starts, time_ends, factors, fps)

            writer.add_scalar(
                f'mlnlp/Progress_Loss',
                loss.item(),
                total_iterations)

            writer.add_scalar(
                f'mlnlp/Progress_Attention_Loss',
                atten_loss.item(),
                total_iterations)

            writer.add_scalar(
                f'mlnlp/Progress_Mean_IoU',
                vis_train.mIoU[-1],
                total_iterations)

            total_iterations += 1.


        writer.add_scalar(
            f'mlnlp/Train_Loss',
            np.mean(vis_train.loss),
            epoch)

        writer.add_scalar(
            f'mlnlp/Train_Mean_IoU',
            np.mean(vis_train.mIoU),
            epoch)

        vis_train.plot(epoch)
        torch.save(model, "./checkpoints/{}/model_epoch_{}".format(cfg.EXPERIMENT_NAME,epoch))

        model.eval()
        for iteration, batch in enumerate(dataloader_test):
            index     = batch[0]

            videoFeat = batch[1].cuda()
            videoFeat_lengths = batch[2].cuda()

            tokens         = batch[3].cuda()
            tokens_lengths = batch[4].cuda()

            start    = batch[5].cuda()
            end      = batch[6].cuda()
            localiz  = batch[7].cuda()
            localiz_lengths = batch[8]
            time_starts = batch[9]
            time_ends = batch[10]
            factors = batch[11]
            fps = batch[12]

            loss, individual_loss, pred_start, pred_end, attention,atten_loss = model(videoFeat, videoFeat_lengths, tokens, tokens_lengths, start, end, localiz)
            vis_test.run(index, pred_start, pred_end, start, end, videoFeat_lengths, epoch, loss.detach(), individual_loss, attention,atten_loss, time_starts, time_ends, factors, fps)
            #print(loss)
            writer.add_scalar(
                f'mlnlp/Progress_Valid_Loss',
                loss.item(),
                total_iterations_val)

            writer.add_scalar(
                f'mlnlp/Progress_Valid_Atten_Loss',
                atten_loss.item(),
                total_iterations_val)

            writer.add_scalar(
                f'mlnlp/Progress_Valid_Mean_IoU',
                vis_test.mIoU[-1],
                total_iterations_val)

            total_iterations_val += 1

        writer.add_scalar(
            f'mlnlp/Valid_Loss',
            np.mean(vis_test.loss),
            epoch)

        writer.add_scalar(
            f'mlnlp/Valid_Mean_IoU',
            np.mean(vis_test.mIoU),
            epoch)

        a = vis_test.plot(epoch)
        writer.add_scalars(f'mlnlp/Valid_tIoU_th', a, epoch)



def tester(cfg):
    print('testing')
    dataloader_test, dataset_size_test   = data.make_dataloader(cfg, is_train=False)

    model = modeling.build(cfg)

    if cfg.TEST.MODEL.startswith('.'):
        load_path = cfg.TEST.MODEL.replace(".", os.path.realpath("."))
    else:
        load_path = cfg.TEST.MODEL

    model = torch.load(load_path)
    model.cuda()

    vis_test  = Visualization(cfg, dataset_size_test, is_train=False)

    writer_path = os.path.join(cfg.VISUALIZATION_DIRECTORY, cfg.EXPERIMENT_NAME)
    writer = SummaryWriter(writer_path)

    total_iterations = 0
    total_iterations_val = 0

    model.eval()
    epoch = 1
    for iteration, batch in enumerate(dataloader_test):
        index     = batch[0]

        videoFeat = batch[1].cuda()
        videoFeat_lengths = batch[2].cuda()

        tokens         = batch[3].cuda()
        tokens_lengths = batch[4].cuda()

        start    = batch[5].cuda()
        end      = batch[6].cuda()
        
        localiz  = batch[7].cuda()
        localiz_lengths = batch[8]
        
        time_starts = batch[9]
        time_ends = batch[10]
        
        factors = batch[11]
        fps = batch[12]

        loss, individual_loss, pred_start, pred_end, attention, atten_loss = model(videoFeat, videoFeat_lengths, tokens, tokens_lengths, start, end, localiz)
        aux = vis_test.run(index, pred_start, pred_end, start, end, videoFeat_lengths, epoch, loss.detach(), individual_loss, attention, atten_loss, time_starts, time_ends, factors, fps)
        total_iterations_val += 1
    a = vis_test.plot(epoch)

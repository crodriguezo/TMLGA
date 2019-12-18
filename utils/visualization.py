import torch
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from utils.miscellaneous import mkdir

class Visualization(object):
    def __init__(self, cfg, dataset_size, is_train=True):

        self.loss = []
        self.IoU  = []
        self.mIoU = []
        self.aux_mIoU = []
        self.individual_loss = {}
        self.vis_dir = "{}{}".format(cfg.VISUALIZATION_DIRECTORY, cfg.EXPERIMENT_NAME)
        print(self.vis_dir)
        mkdir(self.vis_dir)
        self.cfg = cfg
        self.s_samples = np.random.randint(dataset_size, size=4)
        self.s_samples = np.insert(self.s_samples,0, 100)

        for s in self.s_samples:
            self.individual_loss[str(s)] = []
            mkdir("{}/{}".format(self.vis_dir, str(s)))
        if is_train == True:
            self.state = "training"
        else:
            self.state = "testing"

    def tIoU(self, start, end, pred_start, pred_end):
        tt1 = np.maximum(start, pred_start)
        tt2 = np.minimum(end, pred_end)
        # Intersection including Non-negative overlap score.
        segments_intersection = (tt2 - tt1).clip(0)
        # Segment union.
        segments_union = (pred_end - pred_start) \
          + (end - start) - segments_intersection
        # Compute overlap as the ratio of the intersection
        # over union of two segments.
        tIoU = segments_intersection.astype(float) / segments_union
        return tIoU

    def run(self, indexes, pred_start, pred_end, start, end, lengths,
                epoch, loss, individual_loss, attention, atten_loss,
                time_starts, time_ends, factors, fps):
        l = loss.detach().item()
        self.loss.append(l)

        startings = np.argmax(pred_start.detach().cpu().numpy(), axis=1)
        endings   = np.argmax(pred_end.detach().cpu().numpy(), axis=1)
        
        startings = factors * (startings) / fps
        endings = factors * (endings + 1) / fps

        gt_start = np.array(time_starts)
        gt_end   = np.array(time_ends)
        
        iou = self.tIoU(gt_start, gt_end, startings, endings)
        self.IoU.append(iou)
        mIoU = np.mean(iou)
        self.mIoU.append(mIoU)
        ret = {}
        for j, indx in enumerate(indexes):
            ret[int(indx)] = {"iou": round(iou[j], 2),
                        "p_start": round(startings[j], 2),
                        "p_end": round(endings[j], 2)}

        gt_start = np.argmax(start.detach().cpu().numpy(), axis=1)
        gt_end = np.argmax(end.detach().cpu().numpy(), axis=1)
        
        for s_sample in self.s_samples:
            if s_sample in indexes:
                pos = indexes.index(s_sample)
                length = int(lengths[pos].cpu())
                s = gt_start[pos]
                e = gt_end[pos]
                gt__ = np.zeros(length)
                p__  = np.zeros(length)

                gt__[s] = 0.3
                gt__[e] = 0.3

                pred__ = pred_start[pos][:length].detach().cpu().numpy()
                pred__e = pred_end[pos][:length].detach().cpu().numpy()

                p__[np.argmax(pred__)] = 0.3
                p__[np.argmax(pred__e)] = 0.3
                attend__ = attention[pos][:length].detach().cpu().numpy()
                # print(np.max(s))
                # e = end[pos]
                # pred__[s] = 1
                # pred__ = preds[pos][:length].detach().cpu().numpy()
                # gt__   = gts[pos][:length].cpu().numpy()
                # print(pred__.shape, gt__.shape)
                # print(p__)
                plt.figure(figsize=(10,2))
                plt.bar(range(0,length), gt__)
                plt.bar(range(0,length), p__, color='powderblue')
                plt.plot(range(0,length), pred__, color='darkorange')
                plt.plot(range(0,length), pred__e, color='purple')
                plt.plot(range(0,length), attend__, color='green')
                plt.ylim(0, 0.3)
                plt.title("{}".format(self.state))
                plt.savefig("{}/{}/localization_{}.png".format(self.vis_dir, str(s_sample), epoch))
                plt.show()
                plt.close()

                self.individual_loss[str(s_sample)].append(individual_loss[pos].detach())

                plt.figure(figsize=(10,4))
                plt.plot(self.individual_loss[str(s_sample)])
                plt.ylim(bottom=0)
                plt.grid(linestyle='-', linewidth=1)
                plt.savefig("{}/{}/loss_{}.png".format(self.vis_dir, str(s_sample), epoch))
                plt.close()
                # plt.figure(), individual_loss[pos].cpu().item()
        return ret

    def plot(self, epoch):
        plt.figure(figsize=(10,10))
        if self.state == 'training':
            plt.subplot(2, 1, 1)
            plt.plot(self.loss)
            plt.xlabel('Iteration')
            plt.ylabel('loss')
            plt.grid(linestyle='-', linewidth=1)
            plt.ylim(bottom=0)
            plt.subplot(2, 1, 2)
            plt.plot(self.mIoU, color='purple')
            plt.xlabel('Iteration')
            plt.ylabel('mean tIoU')
        else:
            plt.subplot(2, 1, 1)
            plt.plot(self.loss)
            plt.xlabel('Iteration')
            plt.ylabel('loss')
            plt.grid(linestyle='-', linewidth=1)
            plt.ylim(bottom=0)
            plt.subplot(2, 1, 2)
            plt.plot(self.mIoU, color='purple')
            plt.xlabel('Iteration')
            plt.ylabel('mean tIoU')
        plt.ylim(bottom=0)
        plt.grid(linestyle='-', linewidth=1)
        plt.savefig("{}/{}_loss_{}.png".format(self.vis_dir, self.state, epoch))
        plt.show()
        plt.close()
        new_ious = []
        for batch in self.IoU:
            for p in batch:
                new_ious.append(p)
        th = {0.1: 0, 0.2: 0, 0.3: 0, 0.4: 0,0.5: 0, 0.6: 0,0.7: 0,0.8: 0,0.9: 0}
        for i in range(len(new_ious)):
            for k in th.keys():
                if round(new_ious[i],2) >= k:
                    th[k] += 1
        if self.state == "training":
            a = {str(k): round(v * 100 / self.cfg.DATASETS.TRAIN_SAMPLES,2) for k, v in th.items()}
        else:
            a = {str(k): round(v * 100 / self.cfg.DATASETS.TEST_SAMPLES,2) for k, v in th.items()}
        print("################")
        print("{}".format(self.state))
        # print(len(new_ious))
        # print(np.mean(np.array(new_ious)))
        for k, v in a.items():
            print("{}:\t {}".format(k, v))
        print("################")
        
        #np.save("{}/{}_iou_{}.png".format(self.vis_dir, self.state, epoch), self.IoU)
        self.IoU = []
        self.mIoU = []
        return a

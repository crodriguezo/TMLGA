import torch
##


def kl_div(gt, pred, lengths):
    '''
    This function compute the KL divergency between gt localization and
    prediction.

    Parameters
    ----------
    gt:
    pred_local:
    lengths:

    '''
    # pred = torch.softmax(pred, dim=0)
    # print(pred.shape)
    batch_size = int(gt.shape[0])
    individual_loss = []
    for i in range(batch_size):
        length = int(lengths[i].cpu())
        p = pred[i][:length]
        p = torch.softmax(p, dim=0)
        q = gt[i][:length]
        individual_loss.append(torch.sum(p * (p/q).log()))
    total_loss = torch.stack(individual_loss).mean()
    return total_loss, individual_loss, pred

def iou(gt, pred, lengths):
    batch_size = int(gt.shape[0])
    individual_loss = []
    pred = torch.sigmoid(pred)# + 1E-20)
    for i in range(batch_size):
        length = int(lengths[i].cpu())
        g = gt[i][:length]
        p = pred[i][:length]
        I = torch.sum(p*g)
        U = torch.sum(g) + torch.sum(p) - I

        individual_loss.append(1-I/U)
    total_loss = torch.stack(individual_loss).mean()
    return total_loss, individual_loss, pred

def pixel_cross_entropy(gt, pred, lengths):
    batch_size = int(gt.shape[0])
    individual_loss = []
    pred = torch.sigmoid(pred)
    for i in range(batch_size):
        length = int(lengths[i].cpu())
        g = gt[i][:length]
        p = pred[i][:length]
        epsilon = 1E-20

        individual_loss.append(-torch.sum(g * torch.log(p + epsilon) + (1-g) * torch.log((1-p)+epsilon)))
    total_loss = torch.stack(individual_loss).mean()
    return total_loss, individual_loss, pred

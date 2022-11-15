import os
import torch


def save_model(args, model, optimizer, current_epoch):
    model_save_path = os.path.join(args.model_path, args.dataset, args.exp_id)
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    out = os.path.join(model_save_path, "checkpoint_{}.tar".format(current_epoch))
    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': current_epoch}
    torch.save(state, out)

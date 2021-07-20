import time

import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.model_loader import load_model
from utils.evaluate import mean_average_precision


def train(
        train_dataloader,
        query_dataloader,
        retrieval_dataloader,
        arch,
        feature_dim,
        code_length,
        num_classes,
        dynamic_meta_embedding,
        num_prototypes,
        device,
        lr,
        max_iter,
        beta,
        gamma,
        mapping,
        topk,
        evaluate_interval,
):
    """
    Training model.

    Args
        train_dataloader, query_dataloader, retrieval_dataloader(torch.utils.data.dataloader.DataLoader): Data loader.
        arch(str): CNN model name.
        code_length(int): Hash code length.
        device(torch.device): GPU or CPU.
        lr(float): Learning rate.
        max_iter(int): Number of iterations.
        alpha(float): Hyper-parameters.
        topk(int): Compute top k map.
        evaluate_interval(int): Interval of evaluation.

    Returns
        checkpoint(dict): Checkpoint.
    """
    # Load model
    model = load_model(arch, feature_dim, code_length, num_classes, num_prototypes).to(device)

    # Create criterion, optimizer, scheduler
    criterion = LTHNetLoss()
    optimizer = optim.RMSprop(
        model.parameters(),
        lr=lr,
        weight_decay=5e-4,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        max_iter,
        lr / 100,
    )

    # Initialization
    running_loss = 0.
    best_map = 0.
    training_time = 0.
    prototypes = torch.zeros([num_prototypes, feature_dim])
    prototypes = prototypes.to(device)

    # Training
    for it in range(max_iter):
        # update prototypes
        prototypes = generate_prototypes(model, train_dataloader, num_prototypes, feature_dim, device,
                                         dynamic_meta_embedding, prototypes)
        prototypes = prototypes.to(device)

        model.train()
        tic = time.time()
        for data, targets, index in train_dataloader:
            data, targets, index = data.to(device), targets.to(device), index.to(device)
            optimizer.zero_grad()

            #
            hashcodes, assignments, _ = model(data, dynamic_meta_embedding, prototypes)
            loss = criterion(hashcodes, assignments, targets, device, beta, gamma, mapping, it, max_iter)

            running_loss = running_loss + loss.item()
            loss.backward()
            optimizer.step()

        # update step
        scheduler.step()
        training_time = time.time() - tic

        # Evaluate
        if it % evaluate_interval == evaluate_interval - 1:
            # Generate hash code
            query_code, query_assignment = generate_code(model, query_dataloader, code_length, num_classes, device,
                                                         dynamic_meta_embedding, prototypes)
            retrieval_code, retrieval_assignment = generate_code(model, retrieval_dataloader, code_length, num_classes,
                                                                 device,
                                                                 dynamic_meta_embedding,
                                                                 prototypes)

            query_targets = query_dataloader.dataset.get_onehot_targets()
            retrieval_targets = retrieval_dataloader.dataset.get_onehot_targets()

            # Compute map
            mAP = mean_average_precision(
                query_code.to(device),
                retrieval_code.to(device),
                query_targets.to(device),
                retrieval_targets.to(device),
                device,
                topk,
            )

            # # Compute pr curve
            # P, R = pr_curve(
            #     query_code.to(device),
            #     retrieval_code.to(device),
            #     query_targets.to(device),
            #     retrieval_targets.to(device),
            #     device,
            # )

            # Log
            logger.info('[iter:{}/{}][loss:{:.2f}][map:{:.4f}][time:{:.2f}]'.format(
                it + 1,
                max_iter,
                running_loss / evaluate_interval,
                mAP,
                training_time,
            ))
            running_loss = 0.

            # Checkpoint
            if best_map < mAP:
                best_map = mAP

                checkpoint = {
                    'model': model.state_dict(),
                    'qB': query_code.cpu(),
                    'rB': retrieval_code.cpu(),
                    'qL': query_targets.cpu(),
                    'rL': retrieval_targets.cpu(),
                    'qAssignment': query_assignment.cpu(),
                    'rAssignment': retrieval_assignment.cpu(),
                    # 'P': P,
                    # 'R': R,
                    'map': best_map,
                    'prototypes': prototypes.cpu(),
                    'beta': beta,
                    'gamma': gamma,
                    'mapping': mapping,
                }

    return checkpoint


def generate_code(model, dataloader, code_length, num_classes, device, dynamic_meta_embedding, prototypes):
    """
    Generate hash code

    Args
        dataloader(torch.utils.data.dataloader.DataLoader): Data loader.
        code_length(int): Hash code length.
        device(torch.device): Using gpu or cpu.

    Returns
        code(torch.Tensor): Hash code.
    """
    model.eval()
    with torch.no_grad():
        N = len(dataloader.dataset)
        code = torch.zeros([N, code_length])
        assignment = torch.zeros([N, num_classes])
        for data, _, index in dataloader:
            data = data.to(device)
            hash_code, class_assignment, _ = model(data, dynamic_meta_embedding, prototypes)
            code[index, :] = hash_code.sign().cpu()
            assignment[index, :] = class_assignment.cpu()
    torch.cuda.empty_cache()
    return code, assignment


def generate_prototypes(model, dataloader, num_prototypes, feature_dim, device, dynamic_meta_embedding,
                        prototypes_placeholder):
    """
    Generate prototypes (visual memory)

    Args
        dataloader(torch.utils.data.dataloader.DataLoader): Data loader.
        code_length(int): Hash code length.
        device(torch.device): Using gpu or cpu.

    Returns
        code(torch.Tensor): prototypes.
    """
    model.eval()
    with torch.no_grad():
        prototypes = torch.zeros([num_prototypes, feature_dim])
        counter = torch.zeros([num_prototypes])
        for data, targets, _ in dataloader:
            data, targets = data.to(device), targets.to(device)
            _, _, direct_feature = model(data, dynamic_meta_embedding, prototypes_placeholder)
            direct_feature = direct_feature.to('cpu')
            index = torch.nonzero(targets, as_tuple=False)[:, 1]
            index = index.to('cpu')
            for j in range(len(data)):
                prototypes[index[j], :] = prototypes[index[j], :] + direct_feature[j, :]
                counter[index[j]] = counter[index[j]] + 1

        for k in range(num_prototypes):
            prototypes[k, :] = prototypes[k, :] / counter[k]
    torch.cuda.empty_cache()
    return prototypes


class LTHNetLoss(nn.Module):
    """
    LTHNet loss function.

    Args
        epoch (float): the current epoch for calculating alpha (balanced or not).
        beta (float): class-balanced hyper-parameter
        num_per_class mapping: number of samples for each class.
        gamma: cross-entropy-loss vs. class-balanced-loss
    """

    def __init__(self):
        super(LTHNetLoss, self).__init__()
        print('Long-Tailed Hashing Loss works!')

    def forward(self, hashcodes, assignments, targets, device, beta, gamma, mapping, epoch, maxIter):
        # eg. mapping['0']=500, mapping['1']=100, etc.
        # -------------------------------------------------------------
        batch_size = assignments.size(0)
        num_classes = assignments.size(1)
        code_length = hashcodes.size(1)

        # -------------------------------------------------------------
        # mini-batch cross-entropy loss between assignments and targets: softmax-log-NLL-average
        # pointwise loss
        loss_cross_entropy = torch.sum(- torch.log(assignments) * targets) / batch_size

        # balanced factor (class)
        balance_factor = torch.zeros([num_classes])
        for j in range(len(mapping)):
            balance_factor[j] = (1 - beta) / (1 - beta ** mapping[str(j)])
        balance_factor = balance_factor / torch.max(balance_factor)

        # class-balanced loss
        weights = torch.Tensor.repeat(balance_factor, [batch_size, 1]).to(device)
        loss_class_balanced = torch.sum(- torch.log(assignments) * targets * weights) / batch_size

        # gradual learning
        # alpha = 1 - (epoch * 1.0 / maxIter) ** 2

        # overall loss
        # loss = alpha * loss_cross_entropy + (1 - alpha) * (gamma * loss_class_balanced)

        return loss_class_balanced

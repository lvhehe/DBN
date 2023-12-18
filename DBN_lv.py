"""
author:Hehe Lv
date:2022-9-20
"""
import os
import torch
from torch import nn

import matplotlib.pyplot as plt
from DBN.load_data import *
from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from tqdm import tqdm


class Net(nn.Module):
    def __init__(self, time_window, batch_size):
        super(Net, self).__init__()

        self.batchsize = batch_size
        self.time_window = time_window
        self.user_fc = nn.Linear(SERVICE_NUM, USER_FC_FEATURE_SIZE)
        self.service_fc = nn.Linear(USER_NUM, SERVICE_FC_FEATURE_SIZE)
        self.user_pre = nn.Linear(SERVICE_NUM, USER_FC_FEATURE_SIZE)
        self.service_pre = nn.Linear(USER_NUM, SERVICE_FC_FEATURE_SIZE)
        self.lstm = nn.GRU(
            input_size=2 * (USER_FC_FEATURE_SIZE + SERVICE_FC_FEATURE_SIZE),  # user_list
            hidden_size=LSTM_HIDDEN_UNIT,  # rnn hidden unit
            num_layers=LSTM_LAYER_NUM,  # number of rnn layer
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(LSTM_HIDDEN_UNIT, 1)
        self.attention = nn.Linear(self.time_window * 2 * (USER_FC_FEATURE_SIZE + SERVICE_FC_FEATURE_SIZE), 1)
        self.activation = nn.ReLU(inplace=True)
        self.weight = nn.Parameter(torch.Tensor(1, time_window))
        # print(USER_FC_FEATURE_SIZE + SERVICE_FC_FEATURE_SIZE)
        # exit(0)

    def forward(self, batch_time_user_features, user_pre, batch_time_service_features, service_pre):
        # 这里batch*time 统一放在batch的维度，进入全连接层

        user_fc_out = self.activation(self.user_fc(batch_time_user_features))
        user_fc_out = user_fc_out.view(-1, self.time_window, USER_FC_FEATURE_SIZE)

        service_fc_out = self.activation(self.service_fc(batch_time_service_features))
        service_fc_out = service_fc_out.view(-1, self.time_window, SERVICE_FC_FEATURE_SIZE)

        user_pre_out = self.activation(self.user_pre(user_pre))
        user_pre_out = user_pre_out.view(-1, self.time_window, USER_FC_FEATURE_SIZE)

        service_pre_out = self.activation(self.service_pre(service_pre))
        service_pre_out = service_pre_out.view(-1, self.time_window, SERVICE_FC_FEATURE_SIZE)

        lstm_input1 = torch.cat([user_fc_out, service_fc_out], 2)
        lstm_input2 = torch.cat([user_pre_out, service_pre_out], 2)
        lstm_input = torch.cat([lstm_input1, lstm_input2], 2)
        # print(lstm_input.shape)
        weight = self.weight.repeat(user_fc_out.shape[0], 1).reshape(user_fc_out.shape[0], self.time_window, 1)
        # print(weight.shape)

        # lstm_input = weight * lstm_input
        lstm_input = lstm_input  # 不加权
        # print(lstm_input.shape)
        # exit(0)
        # print(attention_fc.shape)

        # attention_fc = attention_fc.view(-1, self.time_window * 2 * (user_fc_out.shape[2] + service_fc_out.shape[2]))
        # attention_out = self.attention(attention_fc)
        # print(attention_fc.shape)
        # exit(0)
        # print(lstm_input.shape)
        # print(attention_fc.shape)
        # print(self.time_window)
        # print(user_fc_out.shape[0])
        # print(user_fc_out.shape[1])
        # print(user_fc_out.shape[2])
        # print(service_fc_out.shape[0])
        # print(service_fc_out.shape[1])
        # print(service_fc_out.shape[2])
        # print(attention_out.shape)
        # exit(0)

        # r_out, (h_n, h_c) = self.lstm(lstm_input, None)  # LSTM时多一个h_c
        r_out, h_n = self.lstm(lstm_input, None)
        # print(r_out)
        # exit(0)

        # print(r_out[:, -1, :].shape)
        # print(exit(0))
        lstm_out = self.out(r_out[:, -1, :])
        # out = lstm_out + attention_out
        # print(out.shape)
        # print(exit(0))

        return lstm_out


def train(trainloader, net, mse_loss, mae_loss, optimizer):
    net.train()
    loss_avg = 0

    for step, (user_batch, user_pre, service_batch, service_pre, target) in enumerate(tqdm(trainloader)):
        # if step > 5:
        #     break
        user_batch = user_batch.view(-1, SERVICE_NUM).cuda()
        user_pre = user_pre.view(-1, SERVICE_NUM).cuda()
        service_batch = service_batch.view(-1, USER_NUM).cuda()
        service_pre = service_pre.view(-1, USER_NUM).cuda()
        target = target.cuda()
        pred = net(user_batch, user_pre, service_batch, service_pre)
        pred = pred.view(-1)

        legal_data_index = torch.ne(target, 20.0) & torch.ne(target, 19.9) & torch.ne(target, 0.0)
        # 防止本批次全部取的0，导致梯度为NaN，再反向传播污染模型中的参数
        if torch.sum(legal_data_index) <= 0:
            continue
        target = target[legal_data_index]
        pred = pred[legal_data_index]

        mse = mse_loss(pred, target)  # 预测结果与最后时刻的结果比较
        mae = mae_loss(pred, target)
        loss = mse + mae
        # n = n + 1
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # back propagation, compute gradients
        optimizer.step()  # apply gradients

        loss_avg += loss.item()

    train_loss = loss_avg / len(trainloader)

    return train_loss
    # writer.add_scalar('title1', train_loss, epoch)  # 可视化变量loss的值


def val(valloader, net, mse_loss, mae_loss):
    # 将模型转为验证模型
    net.eval()
    test_mae, test_rmse, loss_avg = 0.0, 0.0, 0
    with torch.no_grad():
        for step, (user_batch, user_pre, service_batch, service_pre, target) in enumerate(tqdm(valloader)):
            # if step > 5:
            #     break

            user_batch = user_batch.view(-1, SERVICE_NUM).cuda()
            user_pre = user_pre.view(-1, SERVICE_NUM).cuda()
            service_batch = service_batch.view(-1, USER_NUM).cuda()
            service_pre = service_pre.view(-1, USER_NUM).cuda()

            target = target.cuda()
            pred = net(user_batch, user_pre, service_batch, service_pre)
            pred = pred.view(-1)

            legal_data_index = torch.ne(target, 20.0) & torch.ne(target, 19.9) & torch.ne(target, 0.0)
            # 防止本批次全部取的0，导致梯度为NaN，再反向传播污染模型中的参数
            if torch.sum(legal_data_index) <= 0:
                continue
            target = target[legal_data_index]
            pred = pred[legal_data_index]

            mae = mae_loss(pred, target)
            mse = mse_loss(pred, target)  # 预测结果与最后时刻的结果比较
            rmse = torch.sqrt(mse_loss(pred, target))
            loss = mse + mae
            loss_avg += loss.item()
            test_mae += mae.item()
            test_rmse += rmse.item()
        test_mae = test_mae / len(valloader)
        test_rmse = test_rmse / len(valloader)
        test_loss = loss_avg / len(valloader)

    return test_loss, test_mae, test_rmse
    # return test_loss


if __name__ == '__main__':
    # Hyper parameters
    BATCH_SIZE = 64  # 64, 8192
    EPOCH = 20  # 5, 20
    LR = 0.0001  # 0.0001, 0.003 # learning rate
    L2_REGULAR = 0  # 0.0001
    # DENSITY_IDX = 0  # 1,2,3,4,5,6 表示浓度为10%，20%，30% ...
    # TIME_WINDOW = 7
    USE_SIM = True
    USE_BINARY = False

    USER_FC_FEATURE_SIZE = 32
    SERVICE_FC_FEATURE_SIZE = 64
    LSTM_LAYER_NUM = 3
    LSTM_HIDDEN_UNIT = 32

    USER_NUM = 142
    SERVICE_NUM = 4500
    TIME_INTERVAL_TOTAL = 64

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # training_data_file_path = "training_matrix_66_10_20_30_40_50_60_percent.npz"
    # training_data_file_path = "training_matrix_5_10_15_20_percent.npz"
    # test_data_file_path = "test_matrix_5_10_15_20_percent.npz"
    training_data_file_path = "training_matrix(8)_5_10_15_20_percent.npz"
    test_data_file_path = "test_matrix(8)_5_10_15_20_percent.npz"
    # training_data_file_path = "training_matrix_10_20_30_40_50_60_percent.npz"
    # test_data_file_path = "test_matrix_10_20_30_40_50_60_percent.npz"

    service_sim_mat = "service_sim_mat.npz"
    user_sim_mat = "user_sim_mat.npz"
    # sim_mat_file_path = "sim/sim_mat63.npz"

    # TIME_WINDOW = 11
    # DENSITY_IDX = 3
    for TIME_WINDOW in range(0, 10):
        for DENSITY_IDX in range(0, 4):
            # for i in range(1, 9):
            #     TIME_WINDOW = 8*i
            #     DENSITY_IDX = 0
            print("time_window:", TIME_WINDOW, " density_idx:", DENSITY_IDX)

            train_dataset = LoadDatasetWithSim(
                file_path=training_data_file_path,
                # sim_file_path=sim_mat_file_path,
                user_sim_file_path=user_sim_mat,
                service_sim_file_path=service_sim_mat,
                time_window=TIME_WINDOW, density_idx=DENSITY_IDX)
            test_dataset = LoadTestDatasetWithSim(
                file_path=test_data_file_path,
                # training_matrix_path="hole_matrix.npz",
                # testing_data_path="testing_data.npz",
                # sim_file_path=sim_mat_file_path,
                user_sim_file_path=user_sim_mat,
                service_sim_file_path=service_sim_mat,
                time_window=TIME_WINDOW, density_idx=DENSITY_IDX)

            train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
            test_loader = Data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

            net = Net(time_window=TIME_WINDOW, batch_size=BATCH_SIZE)
            net.cuda()  # Moves all model parameters and buffers to the GPU.
            optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=L2_REGULAR)
            mse_loss = nn.MSELoss()  # the target label is not one-hotted
            mae_loss = nn.L1Loss()
            # writer = SummaryWriter('./path/to/log')
            M = 0
            RM = 0
            for epoch in range(EPOCH):
                # break
                print(epoch)
                print('start training...')
                loss = train(train_loader, net, mse_loss, mae_loss, optimizer)
                print('start eval')
                # mae, rmse = val(test_loader, net, mse_loss, mae_loss)
                test_loss, test_mae, test_rmse = val(train_loader, net, mse_loss, mae_loss)
                print(loss)
                print(test_loss)
                print(test_mae)
                print(test_rmse)
                M = test_mae
                RM = test_rmse
            weight = net.weight.data
            print(M)
            print(RM)
            print(weight)
            print("======")

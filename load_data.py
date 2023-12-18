import torch.utils.data as Data
from scipy.stats import pearsonr
import numpy as np
from torch import from_numpy as fn
import torch
import torch.nn as nn


class LoadDataset(Data.Dataset):

    def __init__(self, file_path="training_matrix_66_10_20_30_40_50_60_percent.npz"
                 , time_window=4, density_idx=0):
        """
        training_matrix_path 训练数据文件，读取training_matrix矩阵 （7, 142, 4500, 64）
            density of training set, training_matrix[1] 10%, training_matrix[2] 20%, ...
            默认取training_matrix[0] 66% （全部数据）
        time_window 时间窗口大小，控制往前取多少时刻的数据作为输入特征
        density_idx 控制选择哪个浓度的训练集
        """
        self.time_window = time_window
        self.ust = np.load(file_path)["arr_0"]
        self.ust = self.ust[density_idx, :, :, -time_window:]
        self.ust = np.float32(self.ust)

    def __len__(self):
        return self.ust.shape[0] * self.ust.shape[1]

    def __getitem__(self, idx):
        user_idx = idx // self.ust.shape[1]
        service_idx = idx % self.ust.shape[1]
        user_feature = self.ust[user_idx, :].T
        service_feature = self.ust[:, service_idx].T
        target = self.ust[user_idx, service_idx, -1]
        return fn(user_feature), fn(service_feature), target


class LoadDatasetWithBin(Data.Dataset):

    def __init__(self, file_path="training_matrix_66_10_20_30_40_50_60_percent.npz"
                 , time_window=4, density_idx=1):
        """
        training_matrix_path 训练数据文件，读取training_matrix矩阵 （7, 142, 4500, 64）
            density of training set, training_matrix[1] 10%, training_matrix[2] 20%, ...
            默认取training_matrix[0] 66% （全部数据）
        time_window 时间窗口大小，控制往前取多少时刻的数据作为输入特征
        density_idx 控制选择哪个浓度的训练集
        """
        self.time_window = time_window
        self.ust = np.load(file_path)["arr_0"]
        self.ust = self.ust[density_idx, :, :, -time_window:]
        self.ust = np.float32(self.ust)
        self.ust_binary = np.float32(self.ust > 0)

    def __len__(self):
        return self.ust.shape[0] * self.ust.shape[1]

    def __getitem__(self, idx):
        user_idx = idx // self.ust.shape[1]
        service_idx = idx % self.ust.shape[1]
        user_feature = self.ust_binary[user_idx, :].T
        service_feature = self.ust_binary[:, service_idx].T
        target = self.ust[user_idx, service_idx, -1]
        return fn(user_feature), fn(service_feature), target


class LoadDatasetWithSim(Data.Dataset):

    def __init__(self, file_path="training_matrix_66_10_20_30_40_50_60_percent.npz"
                 , user_sim_file_path="user_sim_mat.npz", service_sim_file_path="service_sim_mat.npz", time_window=4,
                 density_idx=0):
        """
        training_matrix_path 训练数据文件，读取training_matrix矩阵 （7, 142, 4500, 64）
            density of training set, training_matrix[1] 10%, training_matrix[2] 20%, ...
            默认取training_matrix[0] 66% （全部数据）
        sim_file_path 相似矩阵文件
        time_window 时间窗口大小，控制往前取多少时刻的数据作为输入特征
        density_idx 控制选择哪个浓度的训练集
        """
        self.time_window = time_window
        self.ust = np.load(file_path)["arr_0"][density_idx]
        self.ust = self.ust[:, :, -time_window:]
        self.ust = np.float32(self.ust)

        self.user = np.load(user_sim_file_path)["arr_0"]
        self.user = self.user[:, :, -time_window:]
        self.user = np.float32(self.user)

        self.service = np.load(service_sim_file_path)["arr_0"]
        self.service = self.service[:, :, -time_window:]
        self.service = np.float32(self.service)
        # # load similarity feature matrix
        # sim_mat = np.load(sim_file_path)
        # self.user_sim_mat = sim_mat['user_sim_mat']
        # self.service_sim_mat = sim_mat['service_sim_mat']

    def __len__(self):
        return self.user.shape[0] * self.service.shape[1]

    def __getitem__(self, idx):
        user_idx = idx // self.ust.shape[1]
        service_idx = idx % self.ust.shape[1]

        user_feature = self.ust[user_idx, :].copy().T
        service_feature = self.ust[:, service_idx].copy().T
        service_sim_mat1 = self.service[service_idx, :].copy().T
        user_sim_mat1 = self.user[user_idx, :].copy().T

        # # ##计算相似性
        # sim_idex = []
        # for i in range(self.time_window):
        #     user_feature[i, :] = user_feature[i, :] * service_sim_mat1[i, :]
        #     service_feature[i, :] = service_feature[i, :] * user_sim_mat1[i, :]
        #     user_feature[i, :] = np.array(torch.nn.functional.normalize(torch.Tensor(user_feature[i, :]), p=2, dim=0))
        #     service_feature[i, :] = np.array(
        #         torch.nn.functional.normalize(torch.Tensor(service_feature[i, :]), p=2, dim=0))

        for i in range(self.time_window):
            # k = 1/self.time_window  # 线性加权
            # k = 1/np.exp(self.time_window)  # 非线性加权
            k = 1  # 不加权
            # k = sim_idex[i]  # 特征的相似性
            user_feature[i, :] = k * user_feature[i, :]
            service_sim_mat1[i, :] = k*service_sim_mat1[i, :]
            service_feature[i, :] = k * service_feature[i, :]
            user_sim_mat1[i, :] = k*user_sim_mat1[i, :]
        target = self.ust[user_idx, service_idx, -1]
        return fn(user_feature), fn(service_sim_mat1), fn(service_feature), fn(user_sim_mat1), target


class LoadDatasetWithSimBin(Data.Dataset):

    def __init__(self, file_path="training_matrix_66_10_20_30_40_50_60_percent.npz"
                 , user_sim_file_path="user_sim_mat.npz", service_sim_file_path="service_sim_mat.npz", time_window=4,
                 density_idx=0):
        """
        training_matrix_path 训练数据文件，读取training_matrix矩阵 （7, 142, 4500, 64）
            density of training set, training_matrix[1] 10%, training_matrix[2] 20%, ...
            默认取training_matrix[0] 66% （全部数据）
        sim_file_path 相似矩阵文件
        time_window 时间窗口大小，控制往前取多少时刻的数据作为输入特征
        density_idx 控制选择哪个浓度的训练集
        """
        self.time_window = time_window
        self.ust = np.load(file_path)["arr_0"][density_idx]
        self.ust = self.ust[:, :, -time_window:]
        self.ust = np.float32(self.ust)
        self.ust_binary = np.float32(self.ust > 0)
        # load similarity feature matrix
        self.user = np.load(user_sim_file_path)["arr_0"]
        self.user = self.user[:, :, -time_window:]
        self.user = np.float32(self.user)

        self.service = np.load(service_sim_file_path)["arr_0"]
        self.service = self.service[:, :, -time_window:]
        self.service = np.float32(self.service)

    def __len__(self):
        return self.ust.shape[0] * self.ust.shape[1]

    def __getitem__(self, idx):
        user_idx = idx // self.ust.shape[1]
        service_idx = idx % self.ust.shape[1]
        user_feature = self.ust_binary[user_idx, :].copy().T
        service_feature = self.ust_binary[:, service_idx].copy().T
        self.service_sim_mat1 = self.service[service_idx, :].copy().T
        self.user_sim_mat1 = self.user[user_idx, :].copy().T

        for i in range(self.time_window):
            user_feature[i, :] = user_feature[i, :] * self.service_sim_mat1[i, :]
            service_feature[i, :] = service_feature[i, :] * self.user_sim_mat1[i, :]
        target = self.ust[user_idx, service_idx, -1]
        return fn(user_feature), fn(service_feature), target


class LoadDatasetMF(Data.Dataset):

    def __init__(self, file_path="training_matrix_66_10_20_30_40_50_60_percent.npz", density_idx=0):
        """
        读取training_matrix矩阵 （7, 142, 4500, 64）
        density of training set, training_matrix[1] 10%, training_matrix[2] 20%, ...
        默认取training_matrix[0] 66% （全部数据）
        density_idx 控制选择哪个浓度的训练集
        """
        self.ust = np.load(file_path)["arr_0"]
        self.ust = self.ust[density_idx, :, :, -1]
        self.ust = np.float32(self.ust)

    def __len__(self):
        return self.ust.shape[0] * self.ust.shape[1]

    def __getitem__(self, idx):
        user_idx = idx // self.ust.shape[1]
        service_idx = idx % self.ust.shape[1]
        target = self.ust[user_idx, service_idx]
        return user_idx, service_idx, target


# ===================================================
# 以下是加载测试数据集
# ===================================================

class LoadTestDataset(Data.Dataset):

    def __init__(self, training_matrix_path="training_matrix_66_10_20_30_40_50_60_percent.npz"
                 , testing_data_path="testing_data.npz"
                 , time_window=32
                 , density_idx=0):
        """
        training_matrix_path 训练数据文件，读取training_matrix矩阵 （7, 142, 4500, 64）
            density of training set, training_matrix[1] 10%, training_matrix[2] 20%, ...
            默认取training_matrix[0] 66% （全部数据）
        sim_file_path 相似矩阵文件
        testing_data_path 测试集文件
            test_data (142*4500, 3) 每一行 user_id, service_id, qos
        time_window 时间窗口大小，控制往前取多少时刻的数据作为输入特征
        density_idx 控制选择哪个浓度的训练集
        """
        self.training_matrix = np.load(training_matrix_path)["arr_0"]
        self.training_matrix = self.training_matrix[density_idx, :, :, -time_window:]
        self.training_matrix = np.float32(self.training_matrix)
        self.test_data = np.load(testing_data_path)["arr_0"]
        self.test_data = np.float32(self.test_data)

    def __len__(self):
        return self.test_data.shape[0]

    def __getitem__(self, idx):
        user_idx = int(self.test_data[idx, 0])
        service_idx = int(self.test_data[idx, 1])
        user_feature = self.training_matrix[user_idx, :].T
        service_feature = self.training_matrix[:, service_idx].T
        target = self.test_data[idx, 2]
        return fn(user_feature), fn(service_feature), target


class LoadTestDatasetWithBin(Data.Dataset):

    def __init__(self,
                 training_matrix_path="training_matrix_66_10_20_30_40_50_60_percent.npz",
                 testing_data_path="testing_data.npz",
                 time_window=4, density_idx=1):
        """
        training_matrix_path 训练数据文件，读取training_matrix矩阵 （7, 142, 4500, 64）
            density of training set, training_matrix[1] 10%, training_matrix[2] 20%, ...
            默认取training_matrix[0] 66% （全部数据）
        testing_data_path 测试集文件
            test_data (142*4500, 3) 每一行 user_id, service_id, qos
        time_window 时间窗口大小，控制往前取多少时刻的数据作为输入特征
        density_idx 控制选择哪个浓度的训练集
        """
        self.time_window = time_window
        self.ust = np.load(training_matrix_path, )["arr_0"]
        self.ust = self.ust[density_idx, :, :, -time_window:]
        self.ust = np.float32(self.ust)
        self.ust_binary = np.float32(self.ust > 0)
        self.test_data = np.load(testing_data_path)["arr_0"]
        self.test_data = np.float32(self.test_data)

    def __len__(self):
        return self.test_data.shape[0]

    def __getitem__(self, idx):
        user_idx = int(self.test_data[idx, 0])
        service_idx = int(self.test_data[idx, 1])
        user_feature = self.ust_binary[user_idx, :].T
        service_feature = self.ust_binary[:, service_idx].T
        target = self.ust[user_idx, service_idx, -1]
        return fn(user_feature), fn(service_feature), target


class LoadTestDatasetWithSim(Data.Dataset):
    """
    加载测试集，集成相似度
    """

    def __init__(self, file_path="test_matrix_66_10_20_30_40_50_60_percent.npz"
                 , user_sim_file_path="user_sim_mat.npz", service_sim_file_path="service_sim_mat.npz", time_window=4,
                 density_idx=0):
        """
        training_matrix_path 训练数据文件，读取training_matrix矩阵 （7, 142, 4500, 64）
            density of training set, training_matrix[1] 10%, training_matrix[2] 20%, ...
            默认取training_matrix[0] 66% （全部数据）
        sim_file_path 相似矩阵文件
        time_window 时间窗口大小，控制往前取多少时刻的数据作为输入特征
        density_idx 控制选择哪个浓度的训练集
        """
        self.time_window = time_window
        self.ust = np.load(file_path)["arr_0"][density_idx]
        self.ust = self.ust[:, :, -time_window:]
        self.ust = np.float32(self.ust)

        self.user = np.load(user_sim_file_path)["arr_0"]
        self.user = self.user[:, :, -time_window:]
        self.user = np.float32(self.user)

        self.service = np.load(service_sim_file_path)["arr_0"]
        self.service = self.service[:, :, -time_window:]
        self.service = np.float32(self.service)
        # # load similarity feature matrix
        # sim_mat = np.load(sim_file_path)
        # self.user_sim_mat = sim_mat['user_sim_mat']
        # self.service_sim_mat = sim_mat['service_sim_mat']

    def __len__(self):
        return self.user.shape[0] * self.service.shape[1]

    def __getitem__(self, idx):
        user_idx = idx // self.ust.shape[1]
        service_idx = idx % self.ust.shape[1]

        user_feature = self.ust[user_idx, :].copy().T
        service_feature = self.ust[:, service_idx].copy().T
        service_sim_mat1 = self.service[service_idx, :].copy().T
        user_sim_mat1 = self.user[user_idx, :].copy().T

        for i in range(self.time_window):
            user_feature[i, :] = user_feature[i, :] * service_sim_mat1[i, :]
            service_feature[i, :] = service_feature[i, :] * user_sim_mat1[i, :]
        target = self.ust[user_idx, service_idx, -1]
        return fn(user_feature), fn(service_feature), target

    # def __init__(self, training_matrix_path="hole_matrix.npz"
    #              , testing_data_path="testing_data.npz"
    #              , user_sim_file_path="user_sim_mat.npz"
    #              , service_sim_file_path="service_sim_mat.npz"
    #              , time_window=32
    #              , density_idx=0):
    #     """
    #     training_matrix_path 训练数据文件，读取training_matrix矩阵 （7, 142, 4500, 64）
    #         density of training set, training_matrix[1] 10%, training_matrix[2] 20%, ...
    #         默认取training_matrix[0] 66% （全部数据）
    #     sim_file_path 相似矩阵文件
    #     testing_data_path 测试集文件
    #         test_data (142*4500, 3) 每一行 user_id, service_id, qos
    #     time_window 时间窗口大小，控制往前取多少时刻的数据作为输入特征
    #     density_idx 控制选择哪个浓度的训练集
    #     """
    #     self.time_window = time_window
    #     self.training_matrix = np.load(training_matrix_path)["arr_0"]
    #     self.training_matrix = np.float32(self.training_matrix[:, :, -time_window:])
    #     self.test_data = np.load(testing_data_path)["arr_0"]
    #     self.test_data = np.float32(self.test_data)
    #     # load similarity feature matrix
    #     self.user = np.load(user_sim_file_path)["arr_0"]
    #     self.user = self.user[:, :, -time_window:]
    #     self.user = np.float32(self.user)
    #
    #     self.service = np.load(service_sim_file_path)["arr_0"]
    #     self.service = self.service[:, :, -time_window:]
    #     self.service = np.float32(self.service)
    #
    # def __len__(self):
    #     return self.test_data.shape[0]
    #
    # def __getitem__(self, idx):
    #     user_idx = int(self.test_data[idx, 0])
    #     service_idx = int(self.test_data[idx, 1])
    #     user_feature = self.training_matrix[user_idx, :].copy().T
    #     service_feature = self.training_matrix[:, service_idx].copy().T
    #
    #     self.service_sim_mat1 = self.service[service_idx, :].copy().T
    #     self.user_sim_mat1 = self.user[user_idx, :].copy().T
    #
    #     for i in range(self.time_window):
    #         user_feature[i, :] = user_feature[i, :] * self.service_sim_mat1[i, :]
    #         service_feature[i, :] = service_feature[i, :] * self.user_sim_mat1[i, :]
    #     target = self.test_data[idx, 2]
    #     return fn(user_feature), fn(service_feature), target


class LoadTestDatasetWithSimBin(Data.Dataset):
    """
    加载测试集，二值化特征向量、集成相似度
    """

    def __init__(self, training_matrix_path="training_matrix_66_10_20_30_40_50_60_percent.npz"
                 , testing_data_path="testing_data.npz"
                 , user_sim_file_path="user_sim_mat.npz"
                 , service_sim_file_path="service_sim_mat.npz"
                 , time_window=32
                 , density_idx=0):
        """
        training_matrix_path 训练数据文件，读取training_matrix矩阵 （7, 142, 4500, 64）
            density of training set, training_matrix[1] 10%, training_matrix[2] 20%, ...
            默认取training_matrix[0] 66% （全部数据）
        sim_file_path 相似矩阵文件
        testing_data_path 测试集文件
            test_data (142*4500, 3) 每一行 user_id, service_id, qos
        time_window 时间窗口大小，控制往前取多少时刻的数据作为输入特征
        density_idx 控制选择哪个浓度的训练集
        """
        self.time_window = time_window
        self.ust = np.load(training_matrix_path)["arr_0"]
        self.ust = self.ust[density_idx, :, :, -time_window:]
        self.ust = np.float32(self.ust)
        self.ust_binary = np.float32(self.ust > 0)
        self.test_data = np.load(testing_data_path)["arr_0"]
        self.test_data = np.float32(self.test_data)
        # load similarity feature matrix
        self.user = np.load(user_sim_file_path)["arr_0"]
        self.user = self.user[:, :, -time_window:]
        self.user = np.float32(self.user)

        self.service = np.load(service_sim_file_path)["arr_0"]
        self.service = self.service[:, :, -time_window:]
        self.service = np.float32(self.service)

    def __len__(self):
        return self.test_data.shape[0]

    def __getitem__(self, idx):
        user_idx = int(self.test_data[idx, 0])
        service_idx = int(self.test_data[idx, 1])
        user_feature = self.ust_binary[user_idx, :].copy().T
        service_feature = self.ust_binary[:, service_idx].copy().T
        self.service_sim_mat1 = self.service[service_idx, :].copy().T
        self.user_sim_mat1 = self.user[user_idx, :].copy().T

        for i in range(self.time_window):
            user_feature[i, :] = user_feature[i, :] * self.service_sim_mat1[i, :]
            service_feature[i, :] = service_feature[i, :] * self.user_sim_mat1[i, :]
        target = self.test_data[idx, 2]
        return fn(user_feature), fn(service_feature), target


class LoadTestDatasetMF(Data.Dataset):
    def __init__(self, testing_data_path="testing_data.npz"):
        self.test_data = np.load(testing_data_path)["arr_0"]
        self.test_data = np.float32(self.test_data)

    def __len__(self):
        return self.test_data.shape[0]

    def __getitem__(self, idx):
        user_idx = int(self.test_data[idx, 0])
        service_idx = int(self.test_data[idx, 1])
        target = self.test_data[idx, 2]
        return user_idx, service_idx, target

class CORA_ML:
    def __init__(self):
        # the num of samples used for generating Random walkers from the trained model
        self.num_samples = 1000000
        # The length of Random Walk.
        self.rw_len = 16
        # The Batch size
        self.batch_size = 32
        self.lr = 0.0005
        self.max_iterations = 20000
        self.node_num = 2810

        self.noise_mul = None
        self.noise_mul_3 = 2.5206  # eps = 3
        self.noise_mul_1 = 4.6852  # eps = 1
        self.noise_mul_02 = 18.8312  # eps = 0.2

    def select_noise_mul(self, target_epsilon):
        if target_epsilon == 3.0:
            self.noise_mul = self.noise_mul_3
        elif target_epsilon == 1.0:
            self.noise_mul = self.noise_mul_1
        elif target_epsilon == 0.2:
            self.noise_mul = self.noise_mul_02
        else:
            print("Note there is no predefined noise_mul for the current epsilon!")


class CORA_ML_pretrain:
    def __init__(self):
        # The length of Random Walk.
        self.rw_len = 16
        # The Batch size
        self.batch_size = 32
        self.max_iterations = 2000


class DBLP:
    def __init__(self):
        # the num of samples used for generating Random walkers from the trained model
        # Note, for a big num of num_samples (e.g. 100000000), it may use a large memory of GPU
        self.num_samples = 10000000
        # The length of Random Walk.
        self.rw_len = 16
        # The Batch size
        self.batch_size = 256
        self.lr = self.lr = 0.001
        self.max_iterations = 20000
        self.node_num = 16191

        self.noise_mul = None
        self.noise_mul_3 = 3.4292  # eps = 3
        self.noise_mul_1 = 6.49  # eps = 1
        self.noise_mul_02 = 25.8554  # eps = 1

    def select_noise_mul(self, target_epsilon):
        if target_epsilon == 3.0:
            self.noise_mul = self.noise_mul_3
        elif target_epsilon == 1.0:
            self.noise_mul = self.noise_mul_1
        elif target_epsilon == 0.2:
            self.noise_mul = self.noise_mul_02
        else:
            print("Note there is no predefined noise_mul for the current epsilon!")


class DBLP_pretrain:
    def __init__(self):
        # The length of Random Walk.
        self.rw_len = 16
        # The Batch size
        self.batch_size = 256
        self.max_iterations = 2000


class CITESEER:
    def __init__(self):
        # the num of samples used for generating Random walkers from the trained model
        self.num_samples = 1000000
        # The length of Random Walk.
        self.rw_len = 16
        # The Batch size
        self.batch_size = 32
        self.lr = 0.0005
        self.max_iterations = 20000
        self.node_num = 2110

        self.noise_mul = None
        self.noise_mul_3 = 3.2950  # eps = 3
        self.noise_mul_1 = 6.1971  # eps = 1
        self.noise_mul_02 = 25.075  # eps = 0.2

    def select_noise_mul(self, target_epsilon):
        if target_epsilon == 3.0:
            self.noise_mul = self.noise_mul_3
        elif target_epsilon == 1.0:
            self.noise_mul = self.noise_mul_1
        elif target_epsilon == 0.2:
            self.noise_mul = self.noise_mul_02
        else:
            print("Note there is no predefined noise_mul for the current epsilon!")


class CITESEER_pretrain:
    def __init__(self):
        # The length of Random Walk.
        self.rw_len = 16
        # The Batch size
        self.batch_size = 32
        self.max_iterations = 2000


def load_config(dataset_str, pretrain=False):
    if pretrain:
        if 'cora' in dataset_str:
            return CORA_ML_pretrain()
        elif 'citeseer' in dataset_str:
            return CITESEER_pretrain()
        elif 'dblp' == dataset_str:
            return DBLP_pretrain()
        else:
            print("Unknown dataset...")
            exit(1)
    else:
        if 'cora' in dataset_str:
            return CORA_ML()
        elif 'citeseer' in dataset_str:
            return CITESEER()
        elif 'dblp' == dataset_str:
            return DBLP()
        else:
            print("Unknown dataset...")
            exit(1)

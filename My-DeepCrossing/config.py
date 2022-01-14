class Config:
    def __init__(self):
        # self.patho_path = r'D:\desktop\randomForest\data\plas_freq100.csv'
        # self.nonpatho_path = r'D:\desktop\randomForest\data\chrom_freq100.csv'
        # self.patho_path = r'D:\desktop\randomForest\data\k=8_patho_freq.csv'
        # self.nonpatho_path = r'D:\desktop\randomForest\data\k=8_non_freq.csv'

        # self.patho_path = r'D:\desktop\randomForest\data\plas_freq1500.csv'
        self.patho_path = r'E:\models\doc2vec_patho_p500.csv'
        # self.patho_path = r'D:\desktop\randomForest\data\k=8_patho_freq.csv'
        self.nonpatho_path = r'E:\models\doc2vec_nonpatho_p500.csv'

        # self.nonpatho_path = r'D:\desktop\randomForest\data\chrom_freq1500.csv'
        # self.test_path = r'D:\bio4\nonsplit LOO\freq_patho.csv'
        # self.nonpatho_path = r'D:\desktop\randomForest\data\k=8_non_freq.csv'
        # 'embed_dim': 0, # 用于控制稀疏特征经过Embedding层后的稠密特征大小
        # 'min_dim': 256, # 稀疏特征维度小于min_dim的直接进入stack layer，不用经过embedding层
        # 'min_dim': 100000, # 稀疏特征维度小于min_dim的直接进入stack layer，不用经过embedding层
        # self.hidden_layers = [4096, 2048, 1024, 512, 512]
        # self.hidden_layers = [8192, 4096, 2048]
        # self.hidden_layers = [2048, 1024, 512, 512, 64]
        self.hidden_layers = [1024, 512]
        # 'hidden_layers': [256],
        # self.dim_stack = 128000
        self.dim_stack = 128000
        self.num_epoch = 150
        self.batch_size = 1
        self.Dropout = 0.1
        self.lr = 0.0005
        self.l2_regularization = 0.0001
        self.device_id = 0
        # self.use_cuda = True
        self.use_cuda = False
        self.model_name = 'dc.model'
        # self.model_name: 'dc.model'


if __name__ == '__main__':
    config = Config()
    name = config.model_name
    print(name)

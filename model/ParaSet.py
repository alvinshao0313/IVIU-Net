class ParaSet:
    def __init__(self, data_dir, lib_path, lib_symbol, length, width, admm_layers,
                 train_from, batch_size, learning_rate, epoch, save_dir, network, ckpt, date_name):
        self.data_dir = data_dir
        self.lib_path = lib_path
        self.lib_symbol = lib_symbol
        self.length = length
        self.width = width
        self.admm_layers = admm_layers
        self.train_from = train_from
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.save_dir = save_dir
        self.network = network
        self.ckpt = ckpt
        self.date_name = date_name

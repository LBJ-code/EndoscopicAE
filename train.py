from AutoEncoder import AutoEncoder

if __name__ == '__main__':
    auto_encoder = AutoEncoder()
    auto_encoder.train(r"D:\Deep_Learning\MonoDepth2\esophagus\imgs", save_iter_freq=1)
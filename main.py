from ml import HorseOrHumanGAN

if __name__ == "__main__":
    hoh = HorseOrHumanGAN()
    # hoh.train(100)
    hoh.load_weights("autoencoder_weights.hdf5")
    distances = hoh.count_distances("horses_or_humans_test")
    print(distances)

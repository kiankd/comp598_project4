import random
import numpy as np

if __name__ == '__main__':
    num_classes = 15
    samples_per_class = 1200
    total = 15*1200

    train_split = 0.8
    val_split = 0.9
    test_split = 1.0

    train_indexes = []
    val_indexes = []
    test_indexes = []
    
    random.seed(1917)
    for i in range(15):
        r = random.sample(range(samples_per_class), samples_per_class)
        train_indexes += map(lambda x: i*samples_per_class+x, r[: int(samples_per_class*train_split) ])
        val_indexes += map(lambda x: i*samples_per_class+x, r[int(samples_per_class*train_split) : int(samples_per_class*val_split)])
        test_indexes += map(lambda x: i*samples_per_class+x, r[int(samples_per_class*val_split) :])
             
    np.save('../split/train_indexes',np.array(train_indexes))
    np.save('../split/val_indexes',np.array(val_indexes))
    np.save('../split/test_indexes',np.array(test_indexes))



"""
Credit: Some code taken from
https://medium.com/@14prakash/transfer-learning-using-keras-d804b2e04ef8
https://gist.github.com/RyanAkilos/3808c17f79e77c4117de35aa68447045
https://keras.io/applications/#fine-tune-inceptionv3-on-a-new-set-of-classes

"""

from keras.metrics import top_k_categorical_accuracy
from keras.utils import to_categorical
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
from load_dataset import load_split_data
from andnn_util import Timer
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
import pandas as pd
from keras.optimizers import SGD
import os
from math import ceil


def top_3_error(y_true, y_pred):
    return 1 - top_k_categorical_accuracy(y_true, y_pred, k=3)


def top_2_error(y_true, y_pred):
    return 1 - top_k_categorical_accuracy(y_true, y_pred, k=2)


def create_model_using_tinycnn(n_classes, input_shape, input_tensor=None):
    model = Sequential()
    model.add(Convolution2D(32, (3, 3), activation='relu', padding='valid',
                            input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, (3, 3), activation='relu', padding='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, (3, 3), activation='relu', padding='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', top_2_error, top_3_error])
    return model


def create_pretrained_model(n_classes, input_shape, input_tensor=None,
                             base='inceptionv3', base_weights='imagenet'):

    # get the (headless) backbone
    if base == 'resnet50':
        base_model_getter = applications.resnet50.ResNet50
    elif base == 'vgg19':
        base_model_getter = applications.vgg19.VGG19
    elif base == 'inceptionv3':
        base_model_getter = applications.inception_v3.InceptionV3
    else:
        raise ValueError('`base = "%s"` not understood.' % base)
    base_model = base_model_getter(include_top=False,
                                   weights=base_weights,
                                   input_tensor=input_tensor,
                                   input_shape=input_shape,
                                   pooling='avg')

    # put the top back on the model (pooling layer is already included)
    x = base_model.output
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(n_classes, activation="softmax")(x)

    if input_tensor is None:
        inputs = base_model.input
    else:
        inputs = input_tensor

    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                  metrics=['accuracy', top_2_error, top_3_error])
    return model


def main(dataset_dir, base, model_weights, img_shape, testpart, valpart,
         batch_size, epochs, samples_per_class, npy_data=False,
         cifar10=False, top_only_stage=False, augment=True,
         checkpoint_path='checkpoint.h5', test_only=False):

    # warn if arguments conflict
    if augment and (npy_data or cifar10):
        from warnings import warn
        warn("\n\nTo use augmentation, `dataset_dir` must be a directory of "
             "images.  To not get this warning, use to --no_augmentation "
             "flag.\n\n")

    # load data
    if cifar10:
        from keras.datasets import cifar10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train, y_test = to_categorical(y_train), to_categorical(y_test)
        x_val, y_val = x_test, y_test
        # img_shape = x_train.shape[1:]
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']

        # resize data
        def resize(x):
            import numpy as np
            import cv2 as cv
            x_new = np.empty(shape=(x.shape[0],) + tuple(img_shape),
                             dtype=x.dtype)
            dsize = img_shape[:2][::-1]
            for k in range(x.shape[0]):
                x_new[k] = cv.resize(x[k], dsize=dsize)
            x = x_new
            if len(img_shape) == 3 and x.ndim == 3:  # convert to color images
                assert img_shape[2] == 3
                x = np.stack([x] * 3, axis=x.ndim)
            return x
        x_train = resize(x_train)
        x_val = resize(x_val)
        x_test = resize(x_test)
    elif npy_data:
        with Timer('Loading data'):
            x_train, y_train, x_val, y_val, x_test, y_test, class_names = \
                load_split_data(npy_dir=dataset_dir,
                                samples_per_class=samples_per_class,
                                testpart=testpart,
                                valpart=valpart,
                                resize=img_shape,
                                cast='float32',
                                image_shape=(28, 28),
                                flat=True)
        x_train = x_train/255.0
        x_val = x_val/255.0
        x_test = x_test/255.0
        print('x_train.shape:', x_train.shape)
        print('y_train.shape:', y_train.shape)
        print('x_val.shape:', x_val.shape)
        print('y_val.shape:', y_val.shape)
        print('x_test.shape:', x_test.shape)
        print('y_test.shape:', y_test.shape)
    else:
        if augment:
            train_datagen = ImageDataGenerator(
                rescale=1./255, shear_range=0.2, zoom_range=0.2,
                horizontal_flip=True)
        else:
            train_datagen = ImageDataGenerator(rescale=1./255)

        test_datagen = ImageDataGenerator(rescale=1./255)
        train_generator = train_datagen.flow_from_directory(
            os.path.join(dataset_dir, 'train'),
            target_size=img_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            interpolation="lanczos")
        validation_generator = test_datagen.flow_from_directory(
            os.path.join(dataset_dir, 'val'),
            target_size=img_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            interpolation="lanczos")
        _validation_steps = \
            int(ceil(len(validation_generator.classes) / batch_size))
        class_names = [l for l in train_generator.class_indices]

    # create training callbacks for saving checkpoints and early stopping
    checkpoint = ModelCheckpoint(checkpoint_path,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=False,
                                 mode='auto',
                                 period=1)
    early = EarlyStopping(monitor='val_acc',
                          min_delta=0,
                          patience=10,
                          verbose=1,
                          mode='auto')

    # compile model
    if base == 'tinycnn':
        model = create_model_using_tinycnn(n_classes=len(class_names),
                                           input_shape=img_shape,
                                           input_tensor=None)
    else:
        base_weights = None if (model_weights == 'scratch') else 'imagenet'
        model = create_pretrained_model(n_classes=len(class_names),
                                        input_shape=img_shape,
                                        input_tensor=None,
                                        base=base, base_weights=base_weights)
    if model_weights is not None:
        model.load_weights(model_weights)

    # train model
    if not test_only:
        if npy_data or cifar10:
            model.fit(x=x_train,
                      y=y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=1,
                      callbacks=[checkpoint, early],
                      validation_data=(x_val, y_val),
                      shuffle=True,
                      class_weight=None,
                      sample_weight=None,
                      initial_epoch=0,
                      steps_per_epoch=None,
                      validation_steps=None)
        else:
            # stage 1 fine-tuning (top only)
            if top_only_stage:
                top_only_epochs = 3
                print("\nStage 1 training (top only)...\n")
                model.fit_generator(train_generator,
                                    epochs=top_only_epochs,
                                    steps_per_epoch=100,
                                    verbose=1,
                                    callbacks=[checkpoint, early],
                                    validation_data=validation_generator,
                                    validation_steps=_validation_steps,
                                    class_weight=None,
                                    max_queue_size=10,
                                    workers=2,
                                    use_multiprocessing=True,
                                    shuffle=True,
                                    initial_epoch=0)
            else:
                top_only_epochs = 0

            # stage 2 of fine-tuning (last two inception blocks top)
            if base == 'inceptionv3':
                for layer in model.layers[:249]:
                    layer.trainable = False
                for layer in model.layers[249:]:
                    layer.trainable = True
                model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                              loss='categorical_crossentropy',
                              metrics=['accuracy', top_2_error, top_3_error])
                if top_only_stage:
                    print("\nStage 2 training (last two inception "
                          "blocks + top)...\n")
                model.fit_generator(train_generator,
                                    epochs=epochs - top_only_epochs,
                                    steps_per_epoch=100,
                                    verbose=1,
                                    callbacks=[checkpoint, early],
                                    validation_data=validation_generator,
                                    validation_steps=_validation_steps,
                                    class_weight=None,
                                    max_queue_size=10,
                                    workers=2,
                                    use_multiprocessing=True,
                                    shuffle=True,
                                    initial_epoch=top_only_epochs)

    # score over test data
    print("Test Results:\n" + '='*13)
    if npy_data or cifar10:
        y_pred = model.predict(x_test, batch_size).argmax(axis=1)
        metrics = model.evaluate(x=x_test,
                                 y=y_test,
                                 batch_size=batch_size,
                                 verbose=1,
                                 sample_weight=None)
    else:
        y_pred = model.predict_generator(generator=validation_generator,
                                         steps=_validation_steps
                                         ).argmax(axis=1)
        metrics = model.evaluate_generator(generator=validation_generator,
                                           steps=_validation_steps)
    for metric, val in zip(model.metrics_names, metrics):
        print(metric, val)

    # print confusion matrix and scikit-image classification report
    if npy_data or cifar10:
        y_test = y_test.argmax(axis=1)
    else:
        y_test = validation_generator.classes
    cm = pd.DataFrame(confusion_matrix(y_test, y_pred), columns=class_names)
    cm.index = class_names
    print('Confusion Matrix')
    print(cm)
    print('Classification Report')
    print(classification_report(y_test, y_pred, target_names=class_names))


if __name__ == '__main__':
    # parse command line arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_dir",
        help="If --npy invoked, an (unsplit) directory of NPY files.  "
             "Otherwise a split dataset of image files containing two "
             "subdirectories, 'train' and 'test' each containing a "
             "subdirectory for each class.  Use 'cifar10' to use the "
             "cifar10 dataset.")
    parser.add_argument(
        "--no_augmentation", default=False, action='store_true',
        help="Invoke to prevent augmentation.")
    parser.add_argument(
        "--base", default='inceptionv3',
        help="Base model to use. Use 'tinycnn' to train a small CNN from "
             "scratch.")
    parser.add_argument(
        "--npy", default=False, action='store_true',
        help="`dataset_dir` is directory of NPY files.")
    parser.add_argument(
        '--model_weights', default=None,
        help="Model weights (.h5) file to use start with. Omit this flag"
             "to use weights pretrained ImageNet or 'scratch' to "
             "train from scratch.")
    parser.add_argument(
        "--size", default=299, type=int,
        help="Images will be resized to `size` x `size`.")
    parser.add_argument(
        "--channels", default=3, type=int,
        help="Number of channels to assume images will have (usually 1 or 3).")
    parser.add_argument(
        "--batch_size", default=32, type=int,
        help="Training and inference batch size.")
    parser.add_argument(
        "--epochs", default=100, type=int,
        help="Training epochs.")
    parser.add_argument(
        "--samples_per_class", default=10**3, type=int,
        help="Number of samples to include per class (ignored unless "
             "--npy falg invoked).")
    parser.add_argument(
        "--testpart", default=0.1, type=float,
        help="Fraction of data to use for test set.")
    parser.add_argument(
        "--valpart", default=0.1, type=float,
        help="Fraction of data to use for validation.")
    parser.add_argument(
        "--top_only_stage", default=False, action='store_true',
        help="Train for a few epochs on only the top of the model.")
    parser.add_argument(
        "--checkpoint_path", default=None,
        help="Where to save the model weights.  Defaults (roughly speaking) "
             "to '<base_model>-<dataset>.h5'.")
    parser.add_argument(
        "--test_only", default=False, action='store_true',
        help="Where to save the model weights.  Defaults (roughly speaking) "
             "to '<base_model>-<dataset>.h5'.")
    args = parser.parse_args()

    _image_shape = (args.size, args.size, args.channels)

    if args.dataset_dir == 'cifar10':
        args.dataset_dir = None
        _cifar10 = True
    else:
        _cifar10 = False

    if args.checkpoint_path is None:
        if _cifar10:
            args.checkpoint_path = "%s-cifar10.h5" % args.base
        else:
            args.checkpoint_path = \
                "%s-%s.h5" % (args.base, os.path.split(args.dataset_dir)[-1])
    main(dataset_dir=args.dataset_dir,
         base=args.base,
         model_weights=args.model_weights,
         img_shape=_image_shape,
         testpart=args.testpart,
         valpart=args.valpart,
         batch_size=args.batch_size,
         epochs=args.epochs,
         samples_per_class=args.samples_per_class,
         npy_data=args.npy,
         cifar10=_cifar10,
         top_only_stage=args.top_only_stage,
         augment=not args.no_augmentation,
         checkpoint_path=args.checkpoint_path,
         test_only=args.test_only)
    from pprint import pprint
    pprint(vars(args))


    from sys import argv
    # MODEL_WEIGHTS = argv[1]
    # DATASET = argv[2]
    # print(argv)
    #
    # if MODEL_WEIGHTS == 'none':
    #     MODEL_WEIGHTS = None
    # print(MODEL_WEIGHTS)
    # print(DATASET)
    #
    # IMG_SHAPE = (299, 299, 3)  # (75, 75, 3)
    # TEST_PORTION = 0.1
    # VALIDATION_PORTION = 0.1
    # BATCH_SIZE = 32
    # EPOCHS = 50
    # SAMPLES_PER_CLASS = 10**3
    # NPY_DIR = '/home/andy/datasets/quickdraw/NPYs_first10'
    # TEST = False
    # DATASET = ['quickdraw', 'gimages-split', 'gimages-gray-split', 'gimages',
    #            'gimages-gray', 'cifar10'][0]  # split sets are for augmentation
    # BACKBONE = ['inceptionv3', 'tinycnn', 'resnet50', 'vgg19'][0]
    # NUM_CLASSES = len(CLASS_NAMES)
    # # MODEL_WEIGHTS = None
    # # MODEL_WEIGHTS = 'inceptionv3-photos-aug.h5'
    # OUTPUT_ONLY_STAGE = False
    #
    # cifar10_class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
    #                        'dog', 'frog', 'horse', 'ship', 'truck']
    # quickdraw10_class_names = \
    #     ['aircraft carrier', 'airplane',  'alarm clock', 'ambulance',
    #      'angel',  'animal migration', 'ant', 'anvil', 'apple', 'arm']

    # GIMAGES presets
    #     img_shape = (299, 299, 3)  # (224, 224, 3)
    #     samples_per_class = 90
    #     batch_size = 32
    #     CLASS_NAMES = ['aircraft carrier', 'airplane',  'alarm clock', 'ambulance',
    #                    'angel',  'animal migration', 'ant', 'anvil', 'apple', 'arm']


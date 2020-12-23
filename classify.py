import pathlib
import logging
import os
import sys
import common.initial as initial
import random
import tensorflow as tf
import math
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
import matplotlib.pyplot as plt
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def main():
    workdir = os.path.dirname(sys.argv[0])
    if workdir == "." or workdir == "":
        workdir = os.getcwd()
    os.chdir(workdir)

    # debug option
    from optparse import OptionParser
    usage = """ %prog [options] image_root_path test_image_root_path"""
    parser = OptionParser(usage=usage)
    parser.add_option("-X", action="store_true", dest="debug", help="output debug message")
    parser.add_option("-b", action="store", dest="batchsize", type="int", default=64, help="每个装载数据批次的条目数")
    parser.add_option("-e", action="store", dest="total_epochs", type="int", default=30, help="训练的epoch数")
    parser.add_option("-t", action="store_true", dest="test_model", help="只测试模型，使用此参数时，第一个位置参数为模型文件存放路径，第二个位置参数为测试图片路径")
    parser.add_option("-m", action="store", dest="saved_model_path", default="models", help="模型文件存放路径")
    
    (options, input_args) = parser.parse_args()

    if len(input_args) < 1:
        parser.print_help()
        sys.exit(1)

    if not os.path.exists(input_args[0]):
        logging.info("训练图片所在目录%s不存在" %input_args[0])
        sys.exit(1)
    else:
        image_root_path = pathlib.Path(input_args[0])    

    if not os.path.exists(input_args[1]):
        logging.info("测试图片所在目录%s不存在" %input_args[1])
        sys.exit(1)
    else:
        test_image_path = pathlib.Path(input_args[1])
    
    saved_model_path = options.saved_model_path
    if not os.path.isabs(saved_model_path):
        saved_model_path = os.path.join(workdir, options.saved_model_path)
    if not os.path.exists(saved_model_path):
        os.makedirs(saved_model_path)

    BATCH_SIZE = options.batchsize
    img_height = 40
    img_width = 40
    # 设置日志输出格式
    initial.logFormat(options.debug)

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    if not options.test_model:
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            image_root_path,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=BATCH_SIZE)

        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            image_root_path,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=BATCH_SIZE)
        
        class_names = train_ds.class_names

        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        num_classes = len(class_names)

        data_augmentation = keras.Sequential(
        [
            # layers.experimental.preprocessing.RandomFlip("horizontal", 
            #                                            input_shape=(img_height, 
            #                                                        img_width,
            #                                                        3)),
            layers.experimental.preprocessing.RandomCrop(img_height, img_width-1,
                                                        input_shape=(img_height, 
                                                                    img_width,
                                                                    3)),
            layers.experimental.preprocessing.RandomZoom(0.1),
            layers.experimental.preprocessing.RandomRotation(0.2)
        ]
        )

        model = Sequential([
            data_augmentation,
            layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes)
        ])  
        
        model.compile(optimizer=keras.optimizers.Adam(1e-3),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
        
        model.summary()

        epochs=options.total_epochs
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs
        )


        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()
    else:
        if len(input_args) < 2:
            print("使用-t参数时，必须指定模型文件存放路径及测试图片路径")
            parser.print_help()
            sys.exit(1)

    if test_image_path:
        test_image_paths = list(test_image_path.glob('*/*'))
        test_image_paths = [str(path) for path in test_image_paths]

        total_img = len(test_image_paths)
        total_correct = 0

        for test_image_path in test_image_paths:
            img = keras.preprocessing.image.load_img(
                test_image_path, target_size=(img_height, img_width)
            )
            # img_label = label_to_index[pathlib.Path(test_image_path).parent.name]
            img_array = keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0) # Create a batch

            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])

            if class_names[np.argmax(score)] == pathlib.Path(test_image_path).parent.name:
                total_correct += 1
            else:
                print(
                    "This image {} most likely belongs to {} with a {:.2f} percent confidence."
                    .format(test_image_path, class_names[np.argmax(score)], 100 * np.max(score))
                )

        accuracy = 100 * total_correct / total_img
        logging.info("测试图片总数:{}, 预测正确数量:{}， 准确率:{:.2f}".format(total_img, total_correct, accuracy))

if __name__ == '__main__':
    main()
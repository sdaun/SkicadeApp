import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras import optimizers
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from data import get_data, data_preparation
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

######## Basic Model #############
def initialize_basic_model():
    model = Sequential()
    model.add(layers.experimental.preprocessing.Rescaling(1. / 255))
    model.add(layers.Conv2D(16, (3,3), input_shape=(75, 100, 3), padding='same', activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Conv2D(32, (2,2), padding='same', activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(50, activation='relu')) # intermediate layer
    model.add(layers.Dense(7, activation='softmax'))
    return model
  
####### Pretrained models #############
def load_model():
  model = ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3),classes=7)
  return model
def set_nontrainable_layers(model):
    model.trainable = False
    return model
  
def add_last_layers(model):
    base_model = set_nontrainable_layers(model)
    flatten_layer = layers.Flatten()
    dense_layer = layers.Dense(500, activation='relu')
    prediction_layer = layers.Dense(7, activation='softmax')
    dropout_layer = layers.Dropout(0.5)
    model = tf.keras.Sequential([
        base_model,
        dropout_layer,
        flatten_layer,
        dense_layer,
        prediction_layer
    ])
    return model
  
def build_model():
    model = load_model()
    model = add_last_layers(model)
    return model
################### compile & fit #########
def compile_model(model):
    model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy', # "sparse_" allows to avoid one-hot-encoding the target
    metrics = ['accuracy','Recall', 'Precision'])
    return model
def fit_model_val_split(model, X_train_stack, y_train):
    es = EarlyStopping(patience = 10, restore_best_weights = True)
    model.fit(X_train_stack, y_train,
                    validation_split = 0.2,
                    callbacks = [es],
                    epochs = 50,
                    batch_size = 32)
    return model
  
def data_augmentation():
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image
        width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True)  # randomly flip images
    datagen.fit(X_train_stack)
    return datagen
  
def fit_model_data_augmentation_with_val(datagen, model, X_train_stack,X_val_stack, y_train, y_val):
    es = EarlyStopping(patience=10, restore_best_weights=True)
    model.fit_generator(datagen.flow(X_train_stack,y_train, batch_size=32),
                              validation_data = (X_val_stack,y_val),
                              epochs = 35,
                              callbacks = [es])
    return model
  
def fit_model_data_augmentation_without_val(datagen,model, X_train_stack, y_train):
    es = EarlyStopping(patience=10, restore_best_weights=True, monitor = 'loss')
    model.fit_generator(datagen.flow(X_train_stack,y_train, batch_size=32),
                              epochs = 35,
                              callbacks = [es])
    return model
def evaluate_model(X_test_stack, y_test, model):
    print(model.evaluate(X_test_stack, y_test))
if __name__ == '__main__':
    skin_df = get_data(100,75)
    X_train_stack, X_test_stack, y_train, y_test = data_preparation(skin_df, val_set = False)
    model = initialize_basic_model()
    model = compile_model(model)
    model = fit_model_val_split(model, X_train_stack, y_train)
    eval = evaluate_model(X_test_stack, y_test, model)
    print('done')
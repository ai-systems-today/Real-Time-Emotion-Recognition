
#Training of the Emotion Classification Model

from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from processing import load_fer2013
from processing import preprocess_input
from models.cnn import mini_XCEPTION
from sklearn.model_selection import train_test_split
import tensorflow


#Parameters
batch_size = 32
num_epochs = 3
input_shape = (48, 48, 1)
validation_split = .2
verbose = 1
num_classes = 7
patience = 50
base_path = 'models/'


#Data Generator
data_generator = ImageDataGenerator(featurewise_center=False, featurewise_std_normalization=False, rotation_range=10,width_shift_range=0.1, height_shift_range=0.1, zoom_range=.1, horizontal_flip=True)

# Model Parameters/Compilation
model = mini_XCEPTION(input_shape, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()



#Call Backs
log_file_path = base_path + '_emotion_training.log'
csv_logger = CSVLogger(log_file_path, append=False)
early_stop = EarlyStopping('val_loss', patience=patience)
reduce_learningRate = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience/4), verbose=1)
trained_models_path = base_path + '_mini_XCEPTION'
model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1, save_best_only=True)
callbacks = [model_checkpoint, csv_logger, early_stop, reduce_learningRate]



#Loading Datasets
faces, emotions = load_fer2013()
faces = preprocess_input(faces)
num_samples, num_classes = emotions.shape
x_train, x_test, y_train,y_test = train_test_split(faces, emotions,test_size=0.2,shuffle=True)
model.fit_generator(data_generator.flow(x_train, y_train, batch_size), steps_per_epoch=len(x_train) / batch_size, epochs=num_epochs, verbose=1, callbacks=callbacks, validation_data=(x_test,y_test))

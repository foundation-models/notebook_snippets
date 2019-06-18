import keras
from keras.models import model_from_json
from keras.models import model_from_yaml
from sklearn.metrics import roc_auc_score
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping


# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')
    
class Callbacks():
  def __init__(self, model_name, batch_size, epochs):
    prefix = '.epoch-' + str(epochs)
    self.modelCheckpoint = ModelCheckpoint('checkpoints/model.' + model_name + prefix + '.h5', 
                             monitor='val_loss', verbose=0, 
                             save_best_only=True)
    self.csvLogger = CSVLogger('checkpoints/log.' + model_name + prefix + '.csv')
    self.reduceLROnPlateau = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)
    self.tensorboard = TensorBoard(log_dir='checkpoints/logs', batch_size=batch_size)
    self.earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0)

  def getDefaultCallbacks(self):
    return [
      PrintDot(),
      self.csvLogger,
      self.modelCheckpoint,
      self.tensorboard
    ]
    
class Histories(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.aucs = []
        self.losses = []

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        y_pred = self.model.predict(self.model.validation_data[0])
        self.aucs.append(roc_auc_score(self.model.validation_data[1], y_pred))
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


def load_model_json(dir, model_name, epoch):
  return load_model(dir, model_name, epoch, '.json')
def load_model_yaml(dir, model_name, epoch):
  return load_model(dir, model_name, epoch, '.yaml')
def load_model(dir, model_name, epoch, extension):
  # load json and create model
  file_name = dir + '/model.' + model_name + extension
  file = open(file_name, 'r')
  loaded_model = file.read()
  file.close()
  print('load model from file ' + file_name)
  if(extension == '.json'):
    model = model_from_json(loaded_model)
  if(extension == '.yaml'):
    model = model_from_yaml(loaded_model)
  else:
    return 'no valid extension'
  load_model_weights(dir, model, model_name, epoch)
  return model

def save_model_json(dir, model, model_name, epoch):
  save_model(dir, model, model_name, epoch, '.json')
def save_model_yaml(dir, model, model_name, epoch):
  save_model(dir, model, model_name, epoch, '.yaml')
def save_model(dir, model, model_name, epoch, extension):
  # serialize model to JSON
  file_name = dir + '/model.' + model_name + extension
  if(extension == '.json'):
    model_loaded = model.to_json()
  if(extension == '.yaml'):
    model_loaded = model.to_yaml()
  with open(file_name, "w") as file:
      file.write(model_loaded)
  save_model_weights(dir, model, model_name, epoch)
def save_model_weights(dir, model, model_name, epoch):
  if(epoch != 0):
    # serialize weights to HDF5
    file_name = dir + '/model.' + model_name + '.h5'
    model.save_weights(file_name)
  
def load_model_weights(dir, model, model_name, epoch):
  # load serialize weights from HDF
  if(epoch != 0):
    ext = '.epoch' + str(epoch)
    file_name = dir + '/model.' + model_name + ext + '.h5'
    print('loading weights from ', file_name)
    model.load_weights(file_name)
  
print('save and load models from yaml and json files defined.\
 Everything stored in folder ', dir)

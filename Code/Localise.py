import numpy as np
import tensorflow as tf
import keras_tuner as kt
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from argparse import ArgumentParser

def getParams():
    """ Receive user-input of simulation parameters via the command Line interface (CLI) and Python library argparse.
        Default values are provided if no input is specified. 
    
        Returns:
            args: Values defining the knot parameters, of specified type.
    """
    par = ArgumentParser()

    par.add_argument("-p",  "--problem",      type=str,  default="0_5",    help="Options: 0_5 or SQRGRN8 or SQRGRN or GRN8 or SQR8")
    par.add_argument("-d",  "--datatype",     type=str,  default="Writhe", help="Options: 1DWrithe or Writhe or LD or LC or LCW or XYZ")
    par.add_argument("-a",  "--adjacent",     type=bool, default=False,    help="Flag to use adjacent datatype from XYZ")
    par.add_argument("-n",  "--normalised",   type=bool, default=False,    help="Flag to use normalised version of datatype")
    par.add_argument("-t",  "--network",      type=str,  default="FFNN",   help="Type of neural network: FFNN or RNN")

    args = par.parse_args()
    
    return args

def set_constants(problem):
    if problem == "0_5":
        Nconformations = 4000000 # Nconformations = Ntrials*Nknots*Nknotspertrial
        Nknots = 4
        Knotind = ['3_1', '4_1', '5_1', '5_2']

    elif problem == "SQRGRN8":
        Nconformations = 3000000 # Nconformations = Ntrials*Nknots*Nknotspertrial
        Nknots = 3
        Knotind = ['3_1_3_1', '3_1-3_1', '8_20'] # square knot, granny knot, 8_20

    elif problem == "SQRGRN":
        Nconformations = 2000000
        Nknots = 2
        Knotind = ['3_1_3_1', '3_1-3_1'] # square knot, granny knot

    elif problem == "GRN8":
        Nconformations = 2000000
        Nknots = 2
        Knotind = ['3_1-3_1', '8_20'] # granny knot, 8_20 

    elif problem == "SQR8":
        Nconformations = 2000000
        Nknots = 2
        Knotind = ['3_1_3_1', '8_20'] # square knot, 8_20
        
    return Nconformations, Nknots, Knotind

def load():
    print(datatype)
    y_data = np.load("kymodata.npy")
    y_data = np.reshape(y_data, (Nknots*Ntrials*Nknotspertrial, Nbeads))

    if rep == "XYZ":
        data=np.zeros((Nknots*Ntrials*Nknotspertrial,Nbeads*dimensions)) # dataset[conformation][xyzcoords]
        for i in range(Nknots):
            print(i)
            for j in range(Ntrials):
                file = "/storage/datastore-group/cmcs-dmichiel/Joseph_ML_Knots/PoL/100Beads/KNOT" + str(Knotind[i]) + "/TRIAL" + str(j) + "/Analysis/" + str(datatype) + "_KNOT" + str(Knotind[i]) + str(j) + ".dat"
                f = open(file,"r")
                lines=f.readlines()
                frame = int(((i*Ntrials)+j)*Nknotspertrial)
                bead = 0
                for x in lines:
                    if x == '\n':
                        frame += 1
                        bead = 0
                        continue
                    data[frame][bead] = (float(x.split(' ')[0]))
                    data[frame][bead+1] = (float(x.split(' ')[1]))
                    data[frame][bead+2] = (float(x.split(' ')[2]))
                    bead += dimensions
                f.close()    
    elif rep == "LOCAL":
        data=np.zeros((Nknots*Ntrials*Nknotspertrial,Nbeads)) # dataset[conformation][local_metric]
        for i in range(Nknots):
            print(i)
            for j in range(Ntrials):
                file = "/storage/datastore-group/cmcs-dmichiel/Joseph_ML_Knots/PoL/100Beads/KNOT" + str(Knotind[i]) + "/TRIAL" + str(j) + "/Analysis/" + str(datatype) + "_KNOT" + str(Knotind[i]) + str(j) + ".dat"
                f = open(file,"r")
                if datatype == "1DWrithe" or datatype == "Writhe":
                    lines = f.readlines()[1:]
                else:
                    lines = f.readlines()
                frame = int(((i*Ntrials)+j)*Nknotspertrial)
                bead = 0
                for k in range(len(lines)):
                    #skip first empty line and update frame and bead number
                    if (lines[k] == '\n') and (lines[(k+1)%len(lines)] == '\n'):
                        frame += 1
                        bead = 0
                        continue
                    #skip second empty line
                    if (lines[k] == '\n') and (lines[(k+1)%len(lines)] != '\n'):
                        continue
                    data[frame][bead] = (float(lines[k].split(' ')[2]))
                    bead += 1
                f.close()
    print(data.shape)
    print("Checking for empty positions in data array...")
    print(np.where(data==0))
    return data, y_data

def add_column_headers(df):
    columns = []
    
    if rep == "XYZ":
        for i in range(Nbeads):
            columns.append('x'+ str(i))
            columns.append('y' + str(i))
            columns.append('z' + str(i))  
            
    elif rep == "LOCAL":
        for i in range(Nbeads):
            columns.append('Bead'+ str(i))
            
    for i in range(Nbeads):
        columns.append('BeadLabel' +str(i))
    df.columns=columns

    return df

def get_data_splits(df, testsize):

    y_columns = []
    for i in range(Nbeads):
        y_columns.append(-i)

    x = df.drop(df.columns[y_columns], axis=1)
    y = df.iloc[:,y_columns]
    print(x.shape)
    print(y.shape)  

    #delete original dataframe from memory
    del [[df]]
    values,counts = np.unique(y,return_counts=True)
    print("Unique values of label IN DATA SET are: ", values, "with frequency ", counts)

    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=testsize, random_state=42)
    
    if NNTYPE == "RNN":
        x_train = np.array(x_train)
        x_test = np.array(x_test)
        size1 = x_train.shape[0]
        size2 = x_test.shape[0]
        if rep == "XYZ":
            x_train = np.reshape(x_train, (size1, Nbeads*dimensions, 1))
            x_test = np.reshape(x_test, (size2, Nbeads*dimensions, 1))
        elif rep == "LOCAL":
            x_train = np.reshape(x_train, (size1, Nbeads, 1))
            x_test = np.reshape(x_test, (size2, Nbeads, 1))
    
    values,counts = np.unique(y_train,return_counts=True)
    print("Unique values of label IN TRAINING SET are: ", values, "with frequency ", counts)
    return x_train, x_test, y_train, y_test

def setup_RNN(size_input, RNN_hidden_top, hidden_activation, opt):
    model = tf.keras.models.Sequential()

    # add input LSTM layer (input_shape=(size_input,))
    model.add(tf.keras.layers.LSTM(
        RNN_hidden_top[0],
        input_shape=(size_input,1),
        activation=hidden_activation,
        return_sequences=True,
        recurrent_dropout=0.2))
    

    # add bidirectional LSTM layer
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(RNN_hidden_top[1], activation=hidden_activation, return_sequences=True, recurrent_dropout = 0.2)))

    # add intermediate LSTM layers  
    for i in range(len(RNN_hidden_top)-3):
        model.add(tf.keras.layers.LSTM(
            RNN_hidden_top[i+2],
            activation=hidden_activation,
            return_sequences=True,
            recurrent_dropout = 0.2))

    # add final LSTM layer with output only from last memory cell (a la Vandans et al.)
    model.add(tf.keras.layers.LSTM(RNN_hidden_top[-1], activation=hidden_activation))
        
    #final output layer with "Nknots" neurons for the "Nknots" knot types
    model.add(tf.keras.layers.Dense(Nbeads,activation="sigmoid")) 
    
    #loss function compares y_pred to y_true: in this case sparse categoricalcrossentropy
    # used for labels that are integers (CategoricalCrossEntropy used for one-hot encoding)
    loss_fn = tf.keras.losses.BinaryCrossentropy() 

    model.compile(optimizer=opt, #adaptive moment estimation gradient descent
                  #loss="MSE", MSE NOT VALID FOR LABELS THAT DONT REPRESENT KNOT
                  loss=loss_fn,
                  metrics=['accuracy'])

    print(model.summary())

    return model    

def setup_NN(size_input, NN_hidden_top, hidden_activation, opt):
    model = tf.keras.models.Sequential()
#     model.add(tf.keras.layers.Flatten()) #flattens dimensions of data so its 1D and takes as input
    
    # add input layer (input_shape=(size_input,)) and first hidden layer to NN
    model.add(tf.keras.layers.Dense(
        NN_hidden_top[0],
        input_shape=(size_input,),
#         kernel_initializer = tf.keras.initializers.RandomUniform(seed=None),
        activation=hidden_activation))

    # add hidden layers to NN 
    for i in range(len(NN_hidden_top)-1):
        model.add(tf.keras.layers.Dense(
            NN_hidden_top[i+1],
            activation=hidden_activation))

    # add output layer 
    model.add(tf.keras.layers.Dense(
        Nbeads,
        activation='sigmoid'))
    
    # loss function compares y_pred to y_true: in this case sparse categoricalcrossentropy
    # used for labels that are integers (CategoricalCrossEntropy used for one-hot encoding)
    loss_fn = tf.keras.losses.BinaryCrossentropy() 
    
    model.compile(optimizer=opt, #adaptive moment estimation gradient descent
                  #loss="MSE", MSE NOT VALID FOR LABELS THAT DONT REPRESENT KNOT
                  loss=loss_fn,
                  metrics=['accuracy'])
    
    # save and print NN details
    print(model.summary())
    return model

def setup_data_for_training(data, y_data):
    size = data.shape[0]
    if rep == "XYZ":
        data = np.reshape(data,(size,Nbeads*dimensions)) #reshape into row and column format
        
    dfx = pd.DataFrame(data) #make numpy array into pandas
    dfy = pd.DataFrame(y_data)
    df = pd.concat([dfx,dfy], axis=1)
    # df = add_label_column(df) #add y labels column (knot type)
    df = add_column_headers(df) #add column headers for x data
    
    if rep == "XYZ":
        # df = df.sample(frac=0.1).reset_index(drop=True) #shuffle conformations so not grouped by knot type 
        df = df.sample(frac=1).reset_index(drop=True) #shuffle conformations so not grouped by knot type 
    elif rep == "LOCAL":
        df = df.sample(frac=1).reset_index(drop=True) #shuffle conformations so not grouped by knot type

    return df

def train(data, y_data):
    df = setup_data_for_training(data, y_data)    
    x_train,x_test,y_train,y_test = get_data_splits(df,0.1)
    
    if rep == "XYZ":
        size_input = Nbeads*dimensions
    elif rep == "LOCAL":
        size_input = Nbeads
        
    n_epoch = 1000
    if NNTYPE == "FFNN":
        opt = tf.keras.optimizers.Adam(lr=0.001)
        NN_top = [320,320,320,320]
        NN = setup_NN(size_input, NN_top, "relu", opt)
    elif NNTYPE == "RNN":
        opt = tf.keras.optimizers.Adam(lr=0.00001)
        RNN_top = [100,100,100,100]
        NN = setup_RNN(size_input, RNN_top, "relu", opt)
    print("Training...")
    history_data = NN.fit(x_train, y_train, epochs = n_epoch, batch_size = 256, verbose = 1, validation_split=1/5, callbacks=cb_list) 
    y_pred = np.argmax(NN.predict(x_test), axis=-1)
    print(y_pred)
    cm = confusion_matrix(y_test,y_pred)
    print(cm)
    NN.evaluate(x_test,y_test)

    # list all data in history
    # print(history.history.keys())
    plt.plot(history_data.history['accuracy'])
    plt.plot(history_data.history['val_accuracy'])
    # plt.title('Accuracy vs Epoch')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig("MPhysPlotsLOCALISE/acc_trainprocess" + "_" + str(problem) + "_" + str(datatype) + "_" + str(adjacent) + "_" + str(normalised) + "_" + str(NNTYPE) + ".pdf")
    plt.show()
    
    plt.clf()

    plt.plot(history_data.history['loss'])
    plt.plot(history_data.history['val_loss'])
    # plt1.title('Loss vs Epoch')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig("MPhysPlotsLOCALISE/loss_trainprocess" + "_" + str(problem) + "_" + str(datatype) + "_" + str(adjacent) + "_" + str(normalised) + "_" + str(NNTYPE) + ".pdf")
    plt.show()

    return cm

def plot_confusion_matrix(cm,
                          target_names,
                          title,
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """    
#     cm = cm/np.sum(cm)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title("Knot Rep: " + str(rep) + "; Classification: " + str(prob))
    plt.clim(0.0,1.0)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
#     params = {'mathtext.default': 'regular' }          
#     plt.rcParams.update(params)

    plt.tight_layout()
    plt.ylabel('True label')
#     plt.xlabel('Predicted label\nAccuracy={:0.4f}; Misclass={:0.4f}'.format(accuracy, misclass))
    plt.xlabel('Predicted label\nAccuracy={:0.4f}'.format(accuracy))
    plt.savefig("MPhysPlotsLOCALISE/CM-" + str(title) + ".pdf", bbox_inches='tight')
    plt.show()
    return

# def build_model(hp):
#     model = tf.keras.models.Sequential()
    
# #     model.add(tf.keras.layers.Dense(
# #         units = hp.Int('units_0', 32, 512, step=32),
# #         input_shape=(size_input,),
# #         activation="relu"))
    
#     # input layer
#     model.add(tf.keras.layers.Flatten(input_shape=(size_input,)))
    
#     # hidden layers
#     for i in range(hp.Int('layers', 2, 6)):
#         model.add(tf.keras.layers.Dense(units=hp.Int('units_' + str(i), 32, 512, step=32),
#                                     activation="relu"))
   
#     # output layer
#     model.add(tf.keras.layers.Dense(Nknots, activation='softmax'))
    
#     loss_fn = tf.keras.losses.SparseCategoricalCrossentropy() 
#     opt = tf.keras.optimizers.Adam(lr=hp.Choice("lr", values = [.1,.01,.001,.0001,.00001]))
    
#     model.compile(optimizer=opt,
#               loss=loss_fn,
#               metrics=['accuracy'])

#     return model

# def tune(data):
#     df = setup_data_for_training(data)    
#     x_train,x_test,y_train,y_test = get_data_splits(df,0.1)
#     print("Tuning...")
#     tuner1.search(x_train, y_train, epochs=10, batch_size = 256, verbose = 1, validation_split=1/5, callbacks = cb_list)
#     return tuner1

def get_COM(conformation):
    comx = 0
    comy = 0
    comz = 0
    for n in range(Nbeads):
        comx+=conformation[n][0]
        comy+=conformation[n][1]
        comz+=conformation[n][2]
    comx/=Nbeads
    comy/=Nbeads
    comz/=Nbeads
    return comx,comy,comz

def translatebead(conformation, COM):
    x = np.zeros((Nbeads))
    y = np.zeros((Nbeads))
    z = np.zeros((Nbeads))
    for j in range(Nbeads):
        x[j] = conformation[j][0] - COM[0]
        y[j] = conformation[j][1] - COM[1]
        z[j] = conformation[j][2] - COM[2]
    return x,y,z

def get_adjacent(dataset):
    adjacent = np.zeros((len(dataset), Nbeads,dimensions))
    for i in range(len(dataset)):
        for j in range(Nbeads):
            for d in range(dimensions):
                adjacent[i][j][d] = dataset[i][(j+1)%Nbeads][d]-dataset[i][j][d]
    return adjacent


args = getParams()

### SETTING UP KNOT CONTEXT AND CONSTANTS ###

# problem = config.knot['problem']
# datatype = config.knot['datatype']
# adjacent = config.knot['adjacent']
# normalised = config.knot['normalised']

problem = args.problem
datatype = args.datatype
adjacent = args.adjacent
normalised = args.normalised
NNTYPE = args.network

print("Problem: " + str(problem))
print("Datatype: " + str(datatype))
print("Adjacent? " + str(adjacent))
print("Normalised? " + str(normalised))
print("Neural Netowrk: " + str(NNTYPE))

if datatype == "XYZ":
    rep = "XYZ"
else:
    rep = "LOCAL"

Nbeads = 100
dimensions = 3
Ntrials=10
Nknotspertrial=100000
Nconformations, Nknots, Knotind = set_constants(problem)

if rep == "XYZ":
    size_input = Nbeads*dimensions
elif rep == "LOCAL":
    size_input = Nbeads

### TRAINING CALLBACKS ###

# Early Stopping Callback
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, restore_best_weights=True)

# Finding best NN model weights during training process
checkpoint_filepath = 'NN_LOCALISE_model_best' + "_" + str(datatype) + "_" + str(problem) + "_" + str(adjacent) + "_" + str(normalised) + "_" + str(NNTYPE)
mc = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

cb_list=[es,mc]

# # HyperBand algorithm from keras tuner
# tuner1 = kt.Hyperband(
#     build_model,
#     objective='val_accuracy',
#     max_epochs=10,
#     factor=2,
#     directory='KT_HB',
#     project_name=str(datatype)+str(normalised)+str(adjacent)
# )
# # Bayesian Optimisation algorithm from keras tuner
# tuner2 = kt.BayesianOptimization(
#         build_model,
#         objective='val_accuracy',
#         max_trials = 50, 
#         directory='KT_BO', 
#         project_name=str(datatype)+str(normalised)+str(adjacent)
# )

#load datatype into ML format
data, y_data = load()

### EXTRA DATA PREPROCESSING ###
#LD
if datatype == "LD" and normalised == True:
    data = data/Nbeads

#3D/1D WRITHE
if (datatype == "Writhe" and normalised == True) or (datatype == "1DWrithe" and normalised == True):
    #NORMALISING BY THE MAX AND MIN DIFFERENCE VALUES
    normconstant = (np.max(data) - np.abs(np.min(data)))/2
    normfactor = np.max(data) - normconstant
    data = (data-normconstant)/normfactor

#XYZ
if datatype == "XYZ" and adjacent == False:
    data = np.reshape(data, (Nknots*Ntrials*Nknotspertrial, Nbeads, dimensions))
    #Calc COM per conformation using get_COM function
    COM = np.zeros((Nknots*Ntrials*Nknotspertrial,dimensions))
    for i in range(Nknots*Ntrials*Nknotspertrial):
        COM[i][0],COM[i][1], COM[i][2] = get_COM(data[i])

    translatedCOMData = np.zeros((Nknots*Ntrials*Nknotspertrial, Nbeads, dimensions))
    for i in range(Nknots*Ntrials*Nknotspertrial):
            translatedCOMData[i,:,0], translatedCOMData[i,:,1], translatedCOMData[i,:,2] = translatebead(data[i],COM[i])
    data = []
    COM = []
    data = translatedCOMData
    translatedCOMData = []
    data = np.reshape(data, (Nknots*Ntrials*Nknotspertrial, Nbeads*dimensions))

#ADJ and NORMADJ
if datatype == "XYZ" and adjacent == True:
    data = np.reshape(data, (Nknots*Ntrials*Nknotspertrial, Nbeads, dimensions))
    adjacentData = get_adjacent(data)
    data = []
    data = adjacentData
    adjacentData = []
    if normalised == True:
        normconstant = (np.max(data) - np.abs(np.min(data)))/2
        normfactor = np.max(data) - normconstant
        normalisedAdjacent = (data - normconstant)/normfactor
        data = []
        data = normalisedAdjacent
        normalisedAdjacent = []
    data = np.reshape(data, (Nknots*Ntrials*Nknotspertrial, Nbeads*dimensions))

#Train data and get Confusion Matrix 
cm = train(data,  y_data)
title = str(problem) + "_" + str(datatype) + "_" + str(adjacent) + "_" + str(normalised) + "_" + str(NNTYPE)
beads = [i for i in range(100)]
beadclass = [0,1]
plot_confusion_matrix(cm, beadclass, title + "test")
plot_confusion_matrix(cm, beads, title)
print("...Training complete.")
# tuner = tune(data)
# print("...Tuning complete.")


# Clear memory
tf.keras.backend.clear_session()
data = []



import numpy as np
import tensorflow as tf
import numpy as np
from numpy import linalg as la
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt 
import math
import pandas as pd 
import csv

def Convert_to_real(x):

    # Find the frequency spectrum
    X = np.fft.fft(x)
    Xf = np.flip(np.conjugate(X))
    X = np.concatenate((np.zeros(1),X, Xf))
    x = np.real(np.fft.ifft(X))
    return x

def corr2cov(corr, std):
    corr = np.asanyarray(corr)
    std_ = np.asanyarray(std)
    cov = corr * np.outer(std_, std_)
    return cov

def generate_corr_channel(N, corr):
    sigma = [1,1]
    corr = [corr, corr]
    cov = corr2cov(corr, sigma)
    cov[0][0]=1
    cov[1][1]=1
    mean = [0, 0]
    h = np.random.multivariate_normal(mean, cov, N) +  1j*np.random.multivariate_normal(mean, cov, N)
    return h

def data_from_csv(str):
    
    data = []
    with open(str) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            data.append(float(row[0]))
    return(np.array(data))

def training_data(h, M, N):
    
    x = np.random.normal(0, 1, N) + 1j*np.random.normal(0, 1, N)
    Real_x = Convert_to_real(x)
    Real_y = np.zeros((2*N+1,2))
    for i in range(h.shape[1]):
        y = np.convolve(x,h[:,i],mode = 'same')
        Real_y[:,i] = Convert_to_real(y)                        
    
    col = np.concatenate((Real_x[M - 1:], np.zeros((M - 1))))
    row = np.concatenate((Real_x[M - 1::-1], np.zeros(M + 1)))
    x_data = toeplitz(col,row)
    
    y_data = Real_y
        
    return x_data, y_data

def training_data_matlab(M):
    
    x = data_from_csv('Data/datax_1.csv')
    y1 = data_from_csv('Data/labely1_1.csv')
    y2 = data_from_csv('Data/labely2_1.csv')
    
    col = np.concatenate((x[M - 1:], np.zeros(M - 1)))
    row = np.concatenate((x[M - 1::-1], np.zeros(M + 1)))
    x_data = toeplitz(col,row)
    
    y_data = np.zeros((len(y1),2))
    y_data[:,0] = y1
    y_data[:,1] = y2
        
    return x_data, y_data
    

def NN_model(input_shape, L1, L21, L22):
    inputs = tf.keras.Input(shape = (input_shape,), name = 'inputs')
    dense_1 = tf.keras.layers.Dense(units=L1, name = 'dense_1')(inputs)
    dense_21 = tf.keras.layers.Dense(units=L21, activation = tf.nn.relu, name = 'dense_21')(dense_1)
    dense_22 = tf.keras.layers.Dense(units=L22, activation = tf.nn.relu, name = 'dense_22')(dense_1)
    out_1 = tf.keras.layers.Dense(units=1, name = 'out_1')(dense_21)
    out_2 = tf.keras.layers.Dense(units=1, name = 'out_2')(dense_22)
    model = tf.keras.Model(inputs, [out_1, out_2])
    model.summary()
    return model

def plot_history(history):
	plt.figure()
	plt.xlabel('Epoch')
	plt.ylabel('Mean Abs Error')
	plt.plot(history.epoch, 10 * np.log10(np.array(history.history['loss'])), label='Train Loss')
	plt.plot(history.epoch, 10 * np.log10(np.array(history.history['val_loss'])), label='Val Loss')
	plt.legend()
	plt.ylim([-60, 20])
	plt.show()
 
def analyze_results(model, X_test, Y_test):

    yTest1 = Y_test[:,0]
    yTest2 = Y_test[:,1]
    
    yHat = model.predict(X_test)
    yHat1 = [ _[0] for _ in yHat[0]]
    yHat2 = [ _[0] for _ in yHat[1]]
    
    # N = 100
    
    # idx = []
    # idxBad = []
    
    # cnt = 0
    
    # for i in range(N):
    #     if abs(yHat1[i]-yTest1[i]) < abs(yTest1[i]):
    #         idx.append(i)
    #     else:
    #         idxBad.append(i)
    #         cnt = cnt + 1
      
      
    # nom = 0
    # denom = 0
    # for i in idx:
	   #  nom += (yHat1[i]-yTest1[i])**2
	   #  denom += yTest1[i]**2
    
    # # Cancellation without bad predictions
    # print(10*math.log10(nom/denom))

    # Cancellation overall
    print("Cancellation for Ch1: %.2fdB" % (10*math.log10(la.norm(yHat1-yTest1,2)/la.norm(yTest1,2))))
    print("Cancellation for Ch2: %.2fdB" % (10*math.log10(la.norm(yHat2-yTest2,2)/la.norm(yTest2,2))))
      
      

def main():

    # HyperParameters
    N_train = 200000
    N_test = 20000

    N_ch = 16
    M = 32
    corr = 0.975

    L1 = 1             # Layer 1 Size
    L21 = 1
    L22 = 1
    
    
    N_Epochs = 100
    val_split = .2
    batch_size = 400
    loss_weights = [.5, .5]

    # Channel
    h = generate_corr_channel(N_ch, corr)
    X_train, Y_train = training_data(h, M, N_train)
    X_test, Y_test = training_data(h, M, N_test)
    
    # X_train, Y_train = training_data_matlab(M)
    # X_test, Y_test = training_data_matlab(M)

    model = NN_model(X_train.shape[1], L1, L21, L22)

    # Model Compile and train
    model.compile(loss=['mean_squared_error', 'mean_squared_error'],
			  loss_weights = loss_weights,
			  optimizer='adam',
			  metrics=['mean_absolute_error', 'mean_absolute_error'])
    
    history = model.fit(x = X_train, y = [Y_train[:,0], Y_train[:,1]], epochs=N_Epochs, validation_split=val_split, batch_size=batch_size, verbose=0)
    results = model.evaluate(x = X_test, y = [Y_test[:,0], Y_test[:,1]], batch_size = 120, verbose=0)

    # print(results)
    print("Correlation is : %.2f" % (corr))
    print("Cancellation for Ch1: %.2fdB" % (10*math.log10(results[1])))
    print("Cancellation for Ch2: %.2fdB" % (10*math.log10(results[2])))
    analyze_results(model, X_test, Y_test)

    plot_history(history)

    #np.savetxt('Results1/Result_1.txt',history.epoch)
    
    
if __name__ == "__main__":
    main()
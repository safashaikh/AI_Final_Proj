import os
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la

i_train = open(os.path.join('digitdata','trainingimages'),'r')
i_train_label = open(os.path.join('digitdata','traininglabels'),'r')
i_test = open(os.path.join('digitdata','testimages'),'r')
i_test_label = open(os.path.join('digitdata','testlabels'),'r')
i_val = open(os.path.join('digitdata','validationimages'),'r')
i_val_label = open(os.path.join('digitdata','validationlabels'),'r')  

def makelabels(file,num):
    labels = np.zeros(num)
    for i in range(num):
        c = file.read(1)
        labels[i] = int(c)
        c_we = file.read(1)
    return labels
        
def makeimages(file,num,height,width):
    train = np.zeros((num,height,width))
    for k in range(num):
        img = np.zeros((height,width))
        for i in range(height):
            for j in range(width):
                c = file.read(1)
                if c == '#':
                    img[i][j] = 255
                elif c == '+':
                    img[i][j] = 128
            c = file.read(1)
        train[k,:,:] = img
    return train
    
train = makeimages(i_train,5000,28,28)
train_labels = makelabels(i_train_label,5000)
test = makeimages(i_test,1000,28,28)
test_labels = makelabels(i_test_label,1000)
val = makeimages(i_val,1000,28,28)
val_labels = makelabels(i_val_label,1000)

# ------------------ NAIVE BAYES CLASSIFIER --------------------

def NBC(data,labels,testdata):
    numFeat = data.shape[0]
    LabelData = [{'class':x, 'index':i} for i,x in enumerate(labels)] 
    #print(LabelData)
    ClassData = []
    numclasses = 10
    for i in range(numclasses):
        newclass = list(filter(lambda x: x['class']==i, LabelData))
        ClassData.append(newclass)
       
    P = []
    for i in range(numclasses):
        newp = len(ClassData[i])/len(LabelData)
        P.append(newp)
        
    #M = stddata.shape[0]
    probFeat = np.zeros((numclasses,numFeat,3))
    ClassFeat = np.zeros((numclasses,numFeat,3))
           
    for k in range(numclasses):
        for i in range(len(ClassData[k])):
            idx = ClassData[k][i]['index']
            for j in range(numFeat):
                if data[j][idx] == 0:
                    ClassFeat[k][j][0] += 1
                elif data[j][idx] == 128:
                    ClassFeat[k][j][1] += 1
                else:
                    ClassFeat[k][j][2] += 1
                #ClassFeat[k][j].append(features_n[j][idx])
        #print(ClassFeat[k])
        for j in range(numFeat):
            probFeat[k][j][0] = ClassFeat[k][j][0]/len(ClassData[k])
            probFeat[k][j][1] = ClassFeat[k][j][1]/len(ClassData[k])
            probFeat[k][j][2] = ClassFeat[k][j][2]/len(ClassData[k])
            #assert (probFeat[k][j][0] + probFeat[k][j][1] + probFeat[k][j][2]) == 1
        #print(probFeat[k][455])
                  
    predictions = []
    
    for i in range(testdata.shape[1]):
        p = []
        for k in range(numclasses):
            pclass = P[k]
            for j in range(150,640):
                if testdata[j][i] == 0:
                    pclass = pclass * probFeat[k][j][0]
                elif testdata[j][i] == 128:
                    pclass = pclass * probFeat[k][j][1]
                else:
                    pclass = pclass * probFeat[k][j][2]
                #pclass = pclass * probGaussian(features[j][i],meanClass[k][j],stdClass[k][j])
            p.append(pclass)
        predictions.append(p.index(max(p)))
    return predictions

def counterror(pred,labels):
    count = 0;
    for i in range(len(pred)):
        if (pred[i] != int(labels[i])):
            count = count +1
    acc = 1 - count/labels.shape[0]
    return count, acc
    
# ------------------- PERCEPTRON using Sigmoid Activation --------------

NUM_CLASSES = 10

def predict1(row, weights):
    x = weights[0]
    for i in range(len(row)-1):
        x += weights[i + 1] * row[i]
    activation = 1 / (1 + np.exp(-x))
    return activation

# Estimate Perceptron weights using stochastic gradient descent
def train_weights(train, l_rate, n_epoch):
    CORRECT_PRED = True
    weights = np.zeros((NUM_CLASSES,len(train[0])))
    for epoch in range(n_epoch):
        sum_error = 0.0
        for row in train:
            prediction = []
            for i in range(NUM_CLASSES):
                prediction.append(predict1(row, weights[i,:]))
            max_act_fn = max(prediction)
            pred_label = prediction.index(max_act_fn)
            actual_label = row[-1]
            if(pred_label != actual_label):
                CORRECT_PRED = False
                sum_error += 1
            error = np.zeros(NUM_CLASSES)
            for i in range(NUM_CLASSES):
                if (i==pred_label) and (CORRECT_PRED):
                    ground_truth = 1
                    my_pred = 1
                elif (i==pred_label) and (not CORRECT_PRED):
                    ground_truth = 0
                    my_pred = 1
                elif (i==actual_label) and (not CORRECT_PRED):
                    ground_truth = 1
                    my_pred = 0
                else:
                    ground_truth = 0
                    my_pred = 0
                #print("class: "+str(i))
                #print("actual label: " +str(actual_label))
                #print("pred label: " +str(pred_label))
                #print("ground_truth: " +str(ground_truth))
                #print("my_pred: " +str(my_pred))
                error[i] = ground_truth - my_pred
                #print("sum_error: "+str(sum_error))
                weights[i,0] = weights[i,0] + l_rate * error[i]
                for j in range(len(row)-1):
                    weights[i,j + 1] = weights[i,j + 1] + l_rate * error[i] * row[j]
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
    return weights

# Perceptron Algorithm With Stochastic Gradient Descent
def perceptron(train, test, l_rate, n_epoch):
    predictions = []
    weights = train_weights(train, l_rate, n_epoch)
    for row in test:
        pred = []
        for i in range(NUM_CLASSES):
            pred.append(predict(row, weights[i, :]))
        max_act_fn = max(pred)
        pred_label = pred.index(max_act_fn)
        predictions.append(pred_label)
    return(predictions)

def accuracy(expected, predicted):
    error_count = 0
    for i in range(0, len(expected)):
        if(predicted[i]!=expected[i]):
            error_count = error_count + 1
    acc = (len(expected)-error_count)/len(expected)
    return error_count, acc
    
    
import time
from sklearn.metrics import accuracy_score

lr = 0.01
iter = 2000
epochs = 15
NUM_CLASSES = 10


def predict(all_weights, test_images):
    test_images = np.hstack((np.ones((test_images.shape[0], 1)), test_images))
    predicted_labels = np.dot(all_weights, test_images.T)
    predicted_labels = sigmoid(predicted_labels)
    predicted_labels = np.argmax(predicted_labels, axis=0)
    return predicted_labels.T

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def learn(train_images, train_labels, weights):
    tr_size = train_images.shape[0]
    iters = []
    error_values = []
    for i in range(iter):
        h = np.dot(train_images, weights)
        h = sigmoid(h)
        error_value = (np.dot(-1 * train_labels.T, np.log(h)) - np.dot((1 - train_labels).T, np.log(1 - h))) / tr_size
        gradient = np.dot(train_images.T, h - train_labels) / tr_size

        weights = weights - (gradient * lr)

        iters.append(i)
        error_values.append(error_value[0, 0])
    return weights

def train_p(train_images, train_labels):
    # add 1's as x0
    tr_size = train_images.shape[0]
    train_images = np.hstack((np.ones((tr_size, 1)), train_images))
    # add w0 as 0 initially
    all_weights = np.zeros((NUM_CLASSES, train_images.shape[1]))
    train_labels = train_labels.reshape((tr_size, 1))
    train_labels_copy = np.copy(train_labels)

    for j in range(NUM_CLASSES):
        #print("Training Classifier: ", j+1)
        train_labels = np.copy(train_labels_copy)
        # initialize all weights to zero
        weights = np.zeros((train_images.shape[1], 1))
        for k in range(tr_size):
            if train_labels[k, 0] == j:
                train_labels[k, 0] = 1
            else:
                train_labels[k, 0] = 0
        weights = learn(train_images, train_labels, weights)
        all_weights[j, :] = weights.T
    return all_weights


def PC(train_images, train_labels, test_images, test_labels):
    start_time = time.clock()
    print(train_images.shape)
    all_weights = train_p(train_images, train_labels)
    tr_time = time.clock() - start_time
    
    predicted_labels = predict(all_weights, test_images)
    
    accuracy = accuracy_score(test_labels, predicted_labels) * 100

    print("Accuracy is: ", accuracy, "%")
    print("------------------\n")
    return accuracy/100, tr_time

# ------------------- k-NEAREST NEIGHBOR  --------------------------

def KNN(k, features, train_labels, test, test_labels):
    T = test#- mean_i;
    #T = T.transpose()
    features_test = T #Urd.transpose().dot(T)
    #print ("Test set in PCA domain: ",features_test.shape)

    diff = np.zeros(shape=(T.shape[1],features.shape[1]))
    for i in range(T.shape[1]):
        for j in range(features.shape[1]):
            diff[i][j] = la.norm(features_test[:,i]-features[:,j],2)
    #print(diff.shape)
    ## find smallest distance k indices
    ind = np.argsort(diff, axis=1)[:,:k]
    #print(ind.shape)
    newlabels = np.zeros(T.shape[1]);
    errorct = 0
    for i in range(T.shape[1]):
        counts = np.zeros(10)
        label = 0
        for j in range(k):
            for c in range(10):
                if(train_labels[ind[i][j]]==c):
                    counts[c] += 1

        
        label = np.argmax(counts)
        
        if (label != test_labels[i]):
            errorct = errorct + 1
    acc = 1-errorct/T.shape[1]
    return errorct, errorct/T.shape[1], acc

## ------------------- TESTING AND STATS ------------------

def getPercentage(data, labels, percent):
    # data in NxM format
    num = int(percent/100 * data.shape[0])
    #indices = np.random.choice(np.arange(data.shape[0]),num)
    ind = np.arange(data.shape[0])
    np.random.shuffle(ind)
    indices = ind[:num]
    #print(indices)
    sdata = np.array([data[i] for i in indices])
    slabels = np.array([labels[i] for i in indices])
    return sdata, slabels

train = np.array([i.flatten() for i in train])
test = np.array([i.flatten() for i in test]) 
train_MxN = train.transpose()
test_MxN = test.transpose()

#percents = [10,20,30,40,50,60,70,80,90,100]
percents = [20,60,100]

# ======= Naive Bayes ==========
accs_NBC = []
stds_NBC = []
times = []
for i in range(len(percents)):
    acc = np.zeros(5)
    time_ = np.zeros(5)
    for j in range(5):
        c_train, c_labels = getPercentage(train, train_labels, percents[i])
        c_train = c_train.transpose()
        start_time = time.clock()
        pred = NBC(c_train, c_labels, test_MxN)
        time_[j] = time.clock() - start_time
        errorcount, accu = counterror(pred,test_labels)
        acc[j] = accu
        
    print(acc)
    mean = np.mean(acc)
    std = np.std(acc)
    accs_NBC.append(mean)
    stds_NBC.append(std)
    mean_t = np.mean(time_)
    times.append(mean_t)
#print(accs_NBC)
#print(stds_NBC)
fig = plt.figure(1)
ax = fig.add_subplot(111)
plt.plot(percents,accs_NBC)
for xy in zip(percents, accs_NBC):                                       # <--
    ax.annotate('%.2f' % xy[1], xy=xy, textcoords='data')
plt.plot(percents,stds_NBC)
for xy in zip(percents,stds_NBC):                                       # <--
    ax.annotate('%.2f' % xy[1], xy=xy, textcoords='data')
plt.title('DIGITS - Naive Bayes Accuracy and Standard Deviation')
plt.ylabel('value (acc or std)')
plt.xlabel('percent')
plt.legend(['mean accuracy','standard deviation'],loc='upper left')
plt.show()

fig2 = plt.figure(2)
ax = fig2.add_subplot(111)
plt.plot(percents,times)
for xy in zip(percents, times):                                       # <--
    ax.annotate('%.2f' % xy[1], xy=xy, textcoords='data')
plt.title('DIGITS - Naive Bayes Times')
plt.ylabel('average time (seconds)')
plt.xlabel('percent of data')
plt.show()

# ========== Perceptron ================

accs_PC = []
stds_PC = []
times = []
for i in range(len(percents)):
    acc = np.zeros(5)
    time_ = np.zeros(5)
    for j in range(5):
        c_train, c_labels = getPercentage(train, train_labels, percents[i])
        accu, tr_time = PC(c_train, c_labels, test, test_labels)
        time_[j] = tr_time
        acc[j] = accu
    mean = np.mean(acc)
    std = np.std(acc)
    accs_PC.append(mean)
    stds_PC.append(std)
    mean_t = np.mean(time_)
    times.append(mean_t)
fig = plt.figure(2)
ax = fig.add_subplot(111)
plt.plot(percents,accs_PC)
for xy in zip(percents, accs_PC):                                       # <--
    ax.annotate('%.2f' % xy[1], xy=xy, textcoords='data')
plt.plot(percents,stds_PC)
for xy in zip(percents,stds_PC):                                       # <--
    ax.annotate('%.2f' % xy[1], xy=xy, textcoords='data')
plt.title('DIGITS - Perceptron Accuracy and Standard Deviation')
plt.ylabel('value (acc or std)')
plt.xlabel('percent')
plt.legend(['mean accuracy','standard deviation'],loc='upper left')
plt.show()

fig2 = plt.figure(2)
ax = fig2.add_subplot(111)
plt.plot(percents,times)
for xy in zip(percents, times):                                       # <--
    ax.annotate('%.2f' % xy[1], xy=xy, textcoords='data')
plt.title('DIGITS - Perceptron Times')
plt.ylabel('average time (seconds)')
plt.xlabel('percent of data')
plt.show()

# ================ K-NN ==================
accs_KNN = []
stds_KNN = []
times = []
for i in range(len(percents)):
    acc = np.zeros(5)
    time_ = np.zeros(5)
    for j in range(5):
        c_train, c_labels = getPercentage(train, train_labels, percents[i])
        c_train = c_train.transpose()
        k = 7
        start_time = time.clock()
        ec, e , accu = KNN(k, c_train, c_labels, test_MxN, test_labels)
        time_[j] = time.clock() - start_time
        acc[j] = accu
    print(acc)
    mean = np.mean(acc)
    std = np.std(acc)
    accs_KNN.append(mean)
    stds_KNN.append(std)
    mean_t = np.mean(time_)
    times.append(mean_t)
#print(accs_KNN)
#print(stds_KNN)
fig = plt.figure(3)
ax = fig.add_subplot(111)
plt.plot(percents,accs_KNN)
for xy in zip(percents, accs_KNN):                                       # <--
    ax.annotate('%.2f' % xy[1], xy=xy, textcoords='data')
plt.plot(percents,stds_KNN)
for xy in zip(percents,stds_KNN):                                       # <--
    ax.annotate('%.2f' % xy[1], xy=xy, textcoords='data')
plt.title('DIGITS - k-NN Accuracy and Standard Deviation')
plt.ylabel('value (acc or std)')
plt.xlabel('percent')
plt.legend(['mean accuracy','standard deviation'],loc='upper left')
plt.show()

fig2 = plt.figure(2)
ax = fig2.add_subplot(111)
plt.plot(percents,times)
for xy in zip(percents, times):                                       # <--
    ax.annotate('%.2f' % xy[1], xy=xy, textcoords='data')
plt.title('DIGITS - k-NN Times')
plt.ylabel('average time (seconds)')
plt.xlabel('percent of data')
plt.show()

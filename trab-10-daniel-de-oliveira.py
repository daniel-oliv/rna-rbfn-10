#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
import matplotlib.pyplot as plt
import random
import csv


# In[2]:


def toRange(arr, min, max):
    return arr*(max-min)+min;

def randInRange(min, max, shape = 1):
    num = np.random.random(shape)
    return num*(max-min)+min;


# In[3]:


def readCsvNpArray(filename):
    data = np.genfromtxt(filename, delimiter='  ')
    return data


# In[4]:


trainSet = readCsvNpArray(r'../RBF_observacoes.txt')
trainSet


# In[5]:


nIns = trainSet.shape[1]-1 # última coluna é apenas target/y
inputs = trainSet[:,0:nIns]
inputs


# In[6]:


targets = trainSet[:,-1:]
targets


# In[7]:


transposedSet = trainSet.transpose()
transposedSet


# In[8]:


nOuts = 1


# In[9]:


plt.plot(trainSet[:,0],trainSet[:,1],'-o', color='green')
plt.title("X x Y")
plt.show()


# In[10]:


np.sort(np.array([[1,2],[0.5,1]]), axis=0)


# In[11]:


class KMeans(object):

    def __init__(self, k, vecs):
        # matriz de pesos tranposta.
        self.vecs = vecs.copy()
        self.k = k # número de centroids
        self.nInputs = self.vecs.shape[0] # número de entradas N
        self.initCentroids()
        self.train()        
        
    def initCentroids(self):
        #         self.weights = toRange(np.random.rand(nOuts, nIns),1,-1)
        indexes = np.random.randint(0,self.vecs.shape[0],self.k) # same as indexes = np.random.choice(x.shape[0], size=(10,))
#         print('indexes')
#         print(indexes)
        self.centroids = self.vecs[indexes] # advanced indexing always copy the array, so np.copy is optional
#         print('centroids')
#         print(self.centroids)
        
    def getNearestUnit(self, x):
        jmin = 0
        distMin = self.calcDist(x, jmin)
        
        for j in range(self.k):
            dist = self.calcDist(x, j)
            if dist < distMin:
                distMin = dist
                jmin = j
        return jmin
    
    def getCentroidIndexes(self):
        bJmins = np.empty(self.nInputs, dtype=int)
        for i in range(self.nInputs):
            x = self.vecs[i]
            jmin = self.getNearestUnit(x)
            bJmins[i] = jmin
        return bJmins
    
    def calcDist(self, x, j):
        cj = self.centroids[j]       
        distances = (cj-x)**2
        total_dist = np.sum(distances)
        return total_dist
    
    def train(self):
        changed = True
        self.epoch = 0
        mins = self.getCentroidIndexes()
        preMins = np.empty(self.nInputs)
        while changed: #self.epoch <= self.maxEpoch:
#             np.random.shuffle(self.vecs)
            self.epoch += 1
            print('kMeans epoch - ' + str(self.epoch))
            preMins = mins
            
            counts = np.zeros(self.k)
            sums = np.zeros_like(self.centroids)
            for i in range(self.nInputs):
                j = mins[i]
                x = self.vecs[i]
                counts[j] += 1
                sums[j] += x
            for j in range(self.k):
                s = sums[j]
                c = counts[j]
                self.centroids[j] = s/c if c != 0 else self.centroids[j]
            mins = self.getCentroidIndexes()
            #compare next clusters indexes
            changed = not np.array_equal(mins, preMins)
        self.inputCjIndexes = self.getCentroidIndexes()
        ##################!!!!!!!!!!!!!!!! COMMENT THIS FOR MORE DIMENSIONS!!!!!!
        self.centroids = self.centroids[self.centroids[:,0].argsort()]
        print('centroids')
        print(self.centroids)


# In[58]:


kmeans = KMeans(9, inputs) ## try pass the outputs together and implement kMeans++
centroids = kmeans.centroids


# In[59]:


plt.plot(trainSet[:,0],trainSet[:,1],'-o', color='green')
plt.title("X x Y")
plt.plot(centroids[:,0],np.zeros_like(centroids[:,0]),'o', color='red')
plt.title("X x Y")
plt.show()


# In[60]:


def nearestMeanDist(vecs, listLen):
    numVecs = vecs.shape[0]
    meanDists = np.empty(vecs.shape[0])
#     print('vecs')
#     print(vecs)
    for i in range(numVecs):
        vec = vecs[i]
        dists = np.linalg.norm(vec - vecs, axis=1)
#         sortedIndexes = dists.argsort()
#         print('dists')
#         print(dists)
#         print('sortedIndexes')
#         print(sortedIndexes)
#         nearest = vecs[sortedIndexes][:listLen]
#         print('nearest')
#         print(nearest)
        ## já que preciso calcular a média das distâncias dos mais próximos, então já uso distances
        ## somar 1 para tirar o próprio ponto
        dists.sort()
        nearestDists = dists[dists>0][:listLen]
#         print('nearestDists')
#         print(nearestDists)
#         print('dists')
#         print(dists)
        meanDists[i] = nearestDists.mean()
    return meanDists;
        

def calcSds(centroids):# há várias formas de fazer isso -> desvio padrão dos dados, distância entre clusters vizinhos e etc
    # by mean distance of the nearest neighbors centroids -> falam dois vizinhos, mas acho que a dimensão conta demais! 1D basta 1 - o mais próximo
    sds = nearestMeanDist(centroids, 1)
    return sds
    
sds = calcSds(centroids)
print('sds')
print(sds)


# In[61]:


class RBFN(object):

    def __init__(self, nIns, nOuts, learningRate, maxEpoch, centroids, sds):
        self.nIns = nIns
        self.nHidden = centroids.shape[0]
        self.nOuts = nOuts
        # matriz de pesos tranposta.
        self.learningRate = learningRate
        self.maxEpoch = maxEpoch
        self.initWeights()
        self.centroids = centroids # médias das gaussinas
        self.sds = sds # desvio padrão das gaussianas
        
        
    def initWeights(self):
        self.weights = toRange(np.random.rand(nOuts, nHidden),-0.1,0.1) ###! mudar para -0.1 e 0.1 or 0
        self.b = np.zeros(nOuts)
        print('self.weights')
        print(self.weights)
        
    def calcYOut(self, hiddenOut):
#         print('hiddenOut')
#         print(hiddenOut)
#         print('self.weights')
#         print(self.weights)
#         print('np.sum(hiddenOut * self.weights, axis=1)')
#         print(np.sum(hiddenOut * self.weights, axis=1))
#         print('self.b')
#         print(self.b)
        y_ins = np.sum(hiddenOut * self.weights, axis=1) + self.b
#         print('y_ins')
#         print(y_ins)
        ### função de ativação foi a identidade
        ys = y_ins
        return ys
        
        
    def trainStaticFunctionParams(self, inputs, targets):
#         getFuncsOuts = lambda : self.funcsOuts 
        numSamples = inputs.shape[0]
        self.epoch = 0
        self.targets = targets
        self.calcFuncsOut(inputs)
        indexes = np.arange(numSamples)
        self.errors = np.zeros(self.maxEpoch)
#         fig, line1 = self.initGraph()
        
        while self.epoch < self.maxEpoch:
            np.random.shuffle(indexes)
            # TODO - implement batch updating
            currentCentroidsOuts = self.funcsOuts[indexes]
            currentTargets = targets[indexes]
            epoch_error = 0
            for i in range(numSamples):
                z = currentCentroidsOuts[i]
#                 print('z')
#                 print(z)
                t = currentTargets[i]
                ys = self.calcYOut(z)
#                 print('t')
#                 print(t)
                error = (t - ys)
                delWs = self.learningRate * error * z
                delB = self.learningRate * error
#                 print('delWs')
#                 print(delWs)
                self.weights += delWs
                self.b += delB              
                
                ###! keep the outs for better performance
                posYs = self.calcYOut(z)
                posError = (t - posYs)
                sqrError = error**2
                epoch_error += sqrError
            self.errors[self.epoch] = epoch_error
#             self.updateGraph(fig, line1)
#             if self.epoch % 10 == 0:
#                 print('epoch_error')
#                 print(epoch_error)
            
            self.epoch+=1
        self.drawErrorGraph()
        estimations = np.empty(numSamples)
        for i in range(numSamples):
            z = self.funcsOuts[i]
            ys = self.calcYOut(z)
            estimations[i] = ys[0]
        self.drawEstimation(estimations)
    
    def drawErrorGraph(self):
        plt.plot(np.arange(0,self.maxEpoch),self.errors,'-', color='green')
        plt.title("Erro a cada época")
        plt.show()
        
    def drawEstimation(self, estimations):
        print('estimations')
        print(estimations)
        plt.plot(trainSet[:,0],trainSet[:,1],'-o', color='green')
        plt.title("X x Y")
        plt.plot(trainSet[:,0],estimations,'-o', color='darkred')
        plt.show()
    
    def initGraph(self):
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        line1, = ax.plot(np.arange(0,self.maxEpoch),self.errors, 'b-')
        return fig, line1
    
    def updateGraph(self, fig, line1):
        plt.ion()
        line1.set_ydata(self.errors)
        fig.canvas.draw()
    
    def calcFuncsOut(self, inputs):
        self.funcsOuts = np.apply_along_axis(self.calcCentroidsOuts,1, inputs)
        return self.funcsOuts
    
    def calcCentroidsOuts(self, x): ###! add term to normalize probabilities
        return np.exp(-1/(2*sds) * np.sum((x-self.centroids)**2, axis=1))


# In[62]:


nHidden = centroids.shape[0]
learningRate = 0.05
maxEpoch = 200


# In[63]:


net = RBFN(nIns, nOuts, learningRate, maxEpoch, centroids, sds)
net.calcCentroidsOuts(net.centroids[0])


# In[64]:


net.trainStaticFunctionParams(inputs, targets)


# In[ ]:





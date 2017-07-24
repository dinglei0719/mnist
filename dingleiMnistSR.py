import struct
import numpy
import array
import time
import scipy.sparse
import scipy.optimize


def readImages(filePath):
    imageFile = open(filePath, 'rb')
    magic, size, rows, cols = struct.unpack('>IIII', imageFile.read(16))
    images = numpy.fromfile(imageFile, dtype = numpy.uint8)
    images = numpy.reshape(images, (size, rows*cols))
    images = images.astype(numpy.float64)
    return (size, rows, cols, images)
    
def readLabels(filepath):
    labelFile = open(filepath, 'rb')
    magic, size = struct.unpack('>II', labelFile.read(8))
    labels = numpy.fromfile(labelFile, dtype = numpy.uint8)
    return labels
    
def getGroundTruth(labels,classNum):
    grountTruth = numpy.zeros((classNum, len(labels)))
    for i in range(len(labels)):
        grountTruth[labels[i]][i] = 1        
    return grountTruth
    
def costCal(labels, classNum, data, weights):
    lamda = 0.0001
    groundtruthMatrix = getGroundTruth(labels, classNum)
    weights_data = numpy.dot(weights, data)
    hypothesis = numpy.exp(weights_data)
    probabilities = hypothesis / numpy.sum(hypothesis, axis = 0)
    cost = -(numpy.sum(numpy.multiply(groundtruthMatrix, numpy.log(probabilities))) / data.shape[1]) + 0.5 * lamda * numpy.sum(numpy.multiply(weights, weights))
    weightsGrad = (-numpy.dot(groundtruthMatrix - probabilities, numpy.transpose(data))) / data.shape[1] + lamda * weights
    return cost, weightsGrad    
    
    

        
if __name__ == '__main__':    
    print "hello"
    trainSize, rows, cols, trainImage = readImages('./data/train-images-idx3-ubyte')
    imgNum, featureNum = trainImage.shape
    trainLabel = readLabels('./data/train-labels-idx1-ubyte')
    trainImage = trainImage / 255
    classNum = 10
    
    rand = numpy.random.RandomState(int(time.time()))
    weights = 0.005 * numpy.asarray(rand.normal(size = (featureNum * classNum, 1))).reshape(classNum, featureNum)
    
    for i in range(300):
        learningRate = 0.1
        for j in range(i / 100):
            learningRate = learningRate * 0.1
        print i, learningRate        
        cost, weightsGradient = costCal(trainLabel, classNum, trainImage.T, weights)
        weights = weights - learningRate * weightsGradient


#------------test----------------------
    testSize, testRows, testCols, testImage = readImages('./data/t10k-images-idx3-ubyte')
    testImgNum, testFeatureNum = testImage.shape
    testLabel = readLabels('./data/t10k-labels-idx1-ubyte')
    testImage = testImage / 255
    
    weights_test = numpy.dot(weights, testImage.T)
    testHypothesis = numpy.exp(weights_test)
    testProbabilities = testHypothesis / numpy.sum(testHypothesis, axis = 0)
    predictions = numpy.zeros((testImage.shape[0], 1))
    predictions[:, 0] = numpy.argmax(testProbabilities, axis = 0)
    count = 0.0
    for i in range(testImgNum):
        if predictions[i][0] == testLabel[i]:
            count = count + 1
    print """Accuracy :""", count / float(testImgNum)
    
    

#!/usr/bin/env Python
# coding=utf-8

import os 
import json
from random import shuffle,seed,randint
import pdb
from python_speech_features import mfcc
from help_func.feature_transformation import *
import pickle 

def SimpleMfccFeatures(wave_data, samplerate, shift=0.05, featurelength=26):
    temp = mfcc(wave_data, samplerate=samplerate, winlen=0.05, winstep=shift, numcep=featurelength, appendEnergy=False)
    # return stats.zscore(temp)
    b = (temp - np.min(temp)) / np.ptp(temp)
    # b = (b - 0.1307) / 0.3081
    # print(b)
    # print(b.shape)
    return b

CLASS_NUM = 2

# src = os.path.join('E:\\2020Great','liujuzheng','liujuzhengdata','filtered')
src = os.path.join(os.getcwd(),'datasets','filtered')
src1 = os.path.join(src,'fast','20190525')
src2 = os.path.join(src,'fast','20190526')
src3 = os.path.join(src,'fed','20190525')
srcs = [src1,src2,src3]

def seg2data(pi):
    pi_dir = ''
    if 'fas_0525' in pi[3]:
        pi_dir = src1
    elif 'fas_0526' in pi[3]:
        pi_dir = src2
    elif 'fed' in pi[3]:
        pi_dir = src3
    subfolder = [i for i in os.listdir(pi_dir) if pi[3][9:11] in i[-2:]]
    try:
        assert(len(subfolder)==1)
    except:
        print(subfolder)
        pdb.set_trace()
    pi_dir = os.path.join(pi_dir,subfolder[0])
    rage = [i for i in os.listdir(pi_dir) if '.wav' in i]
    target = [os.path.join(pi_dir,i) for i in rage if (pi[3][-5:-3] in i[-13:-10]) and (pi[3][-2:] in i[-10:-6])]
    try:
        assert(len(target) == 1)
    except:
        print(target)
        print(pi_dir)
        pdb.set_trace()
    wavsignal, fs = read_wav_data(target[0])
    pi[0] = float(pi[0])
    pi[1] = float(pi[1])
    try:
        assert( myround(pi[0]*100)%5==0 and myround(pi[1]*100)%5==0 )
    except:
        print(pi[0], ' and ', pi[1])
        pdb.set_trace()
    piece_oned = wavsignal[0,myround(pi[0]*fs):myround(pi[1]*fs)]
    piece = SimpleMfccFeatures(piece_oned, fs) #problematic
    # pdb.set_trace()
    sd = myround((pi[1] - pi[0])*100/5)
    try:
        assert(piece.shape[0] == sd)
    except:
        print('actual piece shape 0: ',piece.shape)
        print ('theoretical value: ',sd)
        pdb.set_trace()
    assert(pi[2] in ('T','F'))
    new = (piece, pi[2],pi[3],(pi[0],pi[1]))
    return new

class dataset_operator():
    def __init__(self,labelpath=None):
        self.labeltable = self.readlabels(labelpath)
        separate, joint = self.dividingNonCross()
        self.trainPieces,self.testPieces, _ = separate
        self.trainList, self.testList = joint
        self.writedataset(joint)
        
    def writedataset(self,tpobject,obpath = None):
        assert(isinstance(tpobject,tuple))
        if obpath is None:
            obpath = os.path.join(os.getcwd(),'datasets','tightdata.pickle')
        with open(obpath,'wb') as f:
            pickle.dump(tpobject,f)
        
    def readlabels(self,jpath=None):
        if jpath is None:
            # jpath = os.path.join('E:\\2020Great', 'tidyBS_dualcodes','mat_codes','new_world.json')
            jpath = os.path.join(os.getcwd(), 'indices','new_world.json')
        else:
            assert(os.path.exists(jpath))
            assert(os.path.isfile(jpath))
        with open(jpath) as json_labels:
            d = json.load(json_labels)
        assert(isinstance(d,dict))
        del d['_TableCols_']
        return d
    
    def dividingNonCross(self, rate=None):
        if rate is None:
            rate = (8,2)
        x = self.labeltable
        Tset = []
        Fset = []
        Eset = [] #E denotes 'exception' set
        num = len(x['_TableRows_'])
        for i in range(num):
            index = x['_TableRows_'][i]
            chunk = x['_TableRecords_'][0][i]
            for el in chunk:
                if el[2] in ('label','file'):
                    continue
                elif el[2] == 'T':
                    tmp = [el[0],el[1],el[2],index]
                    Tset.append(seg2data(tmp))
                elif el[2] == 'F':
                    tmp = [el[0],el[1],el[2],index]
                    Fset.append(seg2data(tmp))
                elif '.wav' in el[2]:
                    tmp = [el[0],el[1],el[2],index]
                    Eset.append(tmp)
                elif isinstance(el,str) and el in ('starting','ending','label','file'):
                    continue
                else:
                    # pdb.set_trace()
                    print('Unresolved data mark: ',el)
                    print('Index: ',index)
                    assert(False)
        seed( 10 )
        shuffle(Tset)
        seed( 10 )
        shuffle(Fset)
        semiTrain = []
        semiTest = []
        trainTrate = myround(len(Tset) * (rate[0]/sum(rate)))
        trainFrate = myround(len(Fset) * (rate[0]/sum(rate)))
        semiTrain = {'T': Tset[0:trainTrate], 'F': Fset[0:trainFrate]}
        semiTest = {'T': Tset[trainTrate:],'F': Fset[trainFrate:]}
        testoverall = []
        trainoverall = []
        for i in ('T', 'F'):
            testoverall = testoverall + semiTest[i]
            trainoverall = trainoverall + semiTrain[i]
        testcnts = self.setCounting(semiTest)
        assert(len(testoverall) == sum(testcnts))
        shuffle(testoverall)
        # print(Eset)
        # pdb.set_trace()
        return (semiTrain, semiTest, Eset),(trainoverall,testoverall)
    
    def setCounting(self,dataset):
        assert(isinstance(dataset,dict))
        assert(all([i in ('T','F') for i in dataset.keys()]))
        return [len(dataset['T']), len(dataset['F'])]
        
    def generatorSetting(self,batch_size=64):
        #1. get the counts of training data
        traincnts = self.setCounting(self.trainPieces)
        assigncnts = [ myround(i/float(sum(traincnts))*batch_size) for i in traincnts]
        if sum(assigncnts) == batch_size:
            pass
        else:
            assert(abs(sum(assigncnts)-batch_size)==1)
            if sum(assigncnts)>batch_size:
                assigncnts[0] = assigncnts[0] - 1
            else:
                assigncnts[0] = assigncnts[0] + 1
        iteration_per_epoch =  min([aggrega//single for aggrega,single in zip(traincnts, assigncnts)])
        classWeights = [sum(traincnts) / (CLASS_NUM * el) for el in traincnts]
        self.sampleoffset = assigncnts
        self.iteration_per_epoch = iteration_per_epoch
        self.classWeights = classWeights  

    def getNonRepetitiveData(self,n_start, type='train', mark=False):
        assert(isinstance(n_start,int))
        assert(type in ('train','test'))
        dataChecked = self.testList if type == 'test' else self.trainList
        num = len(dataChecked)
        data_input, data_label, filepath, stamp = dataChecked[n_start%num]
        maplist = {'T': 1, 'F': 0}
        data_label = maplist[data_label]
        if mark is False:
            return data_input, np.array([data_label])
        else:
            return data_input, np.array([data_label]), filepath, stamp

    def getData(self, n_start):  # Due to the class weight, samples in every batch does not need to be class-equal.
        assert(isinstance(n_start,int))
        data_input = []
        data_label = []
        for tag,offset in zip(('T', 'F'),self.sampleoffset):
            begining = n_start * offset
            temp = self.trainPieces[tag][begining:begining+offset]
            tempunzip = list(map(list,zip(*temp)))
            data_input = data_input + tempunzip[0]
            data_label = data_label + tempunzip[1]
        together = list(zip(data_input,data_label))
        shuffle(together)
        data_input,data_label = zip(*together)
        maplist = { 'T':1, 'F':0}
        data_label = [maplist[i] for i in data_label]
        return data_input, data_label
    
    def data_genetator(self): #This generator is only in charge of one single epoch
        for i in ('T', 'F'):
            seed( 10 )
            shuffle(self.trainPieces[i])
        while True:
            ran_num = randint(0, self.iteration_per_epoch-1)  # 获取一个随机数
            X,y = self.getData(n_start=ran_num)
            X = np.array(X)
            y = np.array(y)
            # yield X, np.eye(CLASS_NUM)[y]
            yield X,y
            # yield X, to_categorical(y, num_classes=CLASS_NUM)  # 功能只是转成独热编码
        pass
        
if __name__ == '__main__':
    shifter = dataset_operator()
    # x = shifter.labeltable
    print(len(shifter.trainList),' and ', len(shifter.testList))
    # shifter.generatorSetting(batch_size=16)
    # yielddata = shifter.data_genetator()
    # for i in yielddata:
        # qq,dd = i
        # print(qq.shape)
        # print(dd)
        # a=1
        # pdb.set_trace()
    
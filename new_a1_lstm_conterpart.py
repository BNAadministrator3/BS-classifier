# 1. read the .grid file
# 2. divide the training and testing files 
# 3. form the training and testing set
# 4. define the network

import tgt
import os
import json
import pdb
from math import floor
def myround(a):
	b=floor(a)
	if (a-b)>=0.5:
		return (b+1)
	else:
		return b

class divdefiles():
    def __init__(self,paths):
        self.paths = paths
        self.train_files, self.test_files = self.dividing(self.paths)
    def dividing(self, paths, rate=None):
        train_files = []
        test_files = []
        if rate is None:
            rate = (8,2)
        for apath in paths:
            subfolders = [os.path.join(apath,i) for i in os.listdir(apath)]
            for sfolder in subfolders:
                if not os.path.isdir(sfolder):
                    print('Its not a folder!')
                    pdb.set_trace()
                else:
                    tgfiles = [os.path.join(sfolder,i) for i in os.listdir(sfolder) if 'TextGrid' in i]
                    train_num = myround(len(tgfiles) * rate[0]/sum(rate))
                    train_files = train_files + tgfiles[0:train_num]
                    test_files = test_files + tgfiles[train_num:]
        return train_files, test_files 
        
    
class get_labels():
    def __init__(self,paths,train_files, test_files):
        self.paths = paths
        train_labels, test_labels = self.dividelabels(train_files,test_files)
        trainlist, testlist = self.setforming(train_files,test_files,train_labels,test_labels)
        # self.overall_labels = {}
        # self.opefile(self.paths)
    
    def read_in_single_textgrid(self, file_name):
        #file_name = os.path.join('/home/zx/Dolphin/Data/textgrid/test/20191011/zcz/1616893', '37429442330632_2019_09_18_01_45_11.TextGrid')
        tg_data = tgt.read_textgrid(file_name) # read a Praat TextGrid file and return a TextGrid object
    #    print(tg_data)
        tier_names = tg_data.get_tier_names()  # get names of all tiers
    #    print (tier_names)
        return tg_data
    
    def extract_interval(self, obj):
        a = []
        for i in range(len(obj)):
            a.append( (obj[i].start_time, obj[i].end_time, obj[i].text) )
        return a
    
    def marking(self,file_path):
        ddir, filename = os.path.split(file_path)
        pre = ''
        if ('fast' in ddir) and ('20190525' in ddir):
            pre = 'fas_0525_'+ddir[-1-1:]
        elif ('fast' in ddir) and ('20190526' in ddir):
            pre = 'fas_0526_'+ddir[-1-1:]
        elif ('fed' in ddir) and ('20190525' in ddir):
            pre = 'fed_0525_'+ddir[-1-1:]
        else:
            assert(0)
        idx = pre + '_' + filename.replace('.','_')
        return idx   
        
    def opefile(self,paths):
        for apath in paths:
            subfolders = [os.path.join(apath,i) for i in os.listdir(apath)]
            for sfolder in subfolders:
                if not os.path.isdir(sfolder):
                    print('Its not a folder!')
                    pdb.set_trace()
                else:
                    tgfiles = [os.path.join(sfolder,i) for i in os.listdir(sfolder) if 'TextGrid' in i]
                    for i in tgfiles:
                        # print(i)
                        tgob = self.read_in_single_textgrid(i)
                        assert(len(tgob.tiers)==1)
                        info = self.extract_interval(tgob.tiers[0].intervals)
                        idx = self.marking(i)
                        self.overall_labels[idx] = info
    
    def dividelabels(self,trainfiles,testfiles):
        train_labels = {}
        test_labels = {}
        for apath in self.paths:
            subfolders = [os.path.join(apath,i) for i in os.listdir(apath)]
            for sfolder in subfolders:
                if not os.path.isdir(sfolder):
                    print('Its not a folder!')
                    pdb.set_trace()
                else:
                    tgfiles = [os.path.join(sfolder,i) for i in os.listdir(sfolder) if 'TextGrid' in i]
                    for i in tgfiles:
                        # print(i)
                        tgob = self.read_in_single_textgrid(i)
                        assert(len(tgob.tiers)==1)
                        info = self.extract_interval(tgob.tiers[0].intervals)
                        idx = self.marking(i)
                        if i in trainfiles:
                            train_labels[idx] = info
                        elif i in testfiles:
                            test_labels[idx] = info
                        else:
                            print('Bad recognition')
                            pdb.set_trace()
        return train_labels, test_labels

    def setforming(train_files,test_files,train_labels,test_labels):
        trainlist = []
        trainTset = []
        trainFset = []
        testlist = []
        testTset = []
        testFset = []
        for apath in self.paths:
            subfolders = [os.path.join(apath,i) for i in os.listdir(apath)]
            for sfolder in subfolders:
                if not os.path.isdir(sfolder):
                    print('Its not a folder!')
                    pdb.set_trace()
                else:
                    wavfiles = [os.path.join(sfolder,i) for i in os.listdir(sfolder) if '.wav' in i]
                    for i in wavfiles:
                        similar_name = wavfiles.replace('.wav','.TextGrid')
                        itsid = self.marking(similar_name)
                        if similar_name in train_files:
                            labeltable = train_labels[itsid]
                        elif similar_name in test_files:
                            labeltable = test_files[itsid]
                        else:
                            print('Wrong decision')
                            pdb.set_trace()
                        #2. check labeltable
                        for item in labeltable:
                            if item[2] == 'T':
                                cha = item[1] - item[0]
                                
                                
                        
        

if __name__ == '__main__':
    src = os.path.join('/home/zhaok14/codes/tidyBS/datasets/filtered/')
    src1 = os.path.join(src,'fast','20190525')
    src2 = os.path.join(src,'fast','20190526')
    src3 = os.path.join(src,'fed','20190525')
    srcs = [src1,src2,src3]
    s = divdefiles(srcs)
    a = get_labels(srcs,s.train_files,s.test_files)
    pdb.set_trace()
    
    
    
    
    
    
    
    
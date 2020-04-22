from A1c_leaveOneOutData import dataset_operator,  buildTrainValtest
import pdb
import matplotlib.pyplot as plt
import numpy as np

opit = dataset_operator()
data = buildTrainValtest(opit.datatable, opit.labeltable)
# its temporary
batch_size = 64
for i in range(len(opit.datatable)):
    data.cutintopieces(flag=i)
    data.generatorSetting(batch_size=batch_size)
    num_data = len(data.testList)
    file_name = opit.datatable[i][0]
    for j in range(num_data):
        datum, _, timestamp, label = data.getNonRepetitiveData( j % num_data, type='test', mark=True) 
        image = np.rot90(datum[0])
        plt.imshow(image,cmap = 'gray_r')
        plt.show()
        pdb.set_trace()
        
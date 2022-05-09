import os
import shutil
scene_list = ['00'+str(int(i)) for i in range(10)] + ['0'+str(int(i)) for i in range(10,30)]
file_list = ['0000000'+str(int(i)) for i in range(10)] + ['000000'+str(int(i)) for i in range(10,100)]
for scene in scene_list:
    for file in file_list:
        shutil.copy('val_sharp/'+scene+'/'+file+'.png', 'val_sharp_2/'+scene+'_'+file[-3:]+'.png')

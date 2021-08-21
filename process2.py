from os import listdir,mkdir
from utity import Ground_truth
import numpy as np
import random
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

class Gesture_Data():
    def __init__(self,path,windows_size=128,gesture_label=[-1000,-1000,-1000]):
        self.path = path
        self.accomplish_path = None
        self.data_classes = list()
        self.data_classes_total = 0
        # self.sigma = gaussian_sigma
        self.windows_size = windows_size
        self.gesture_label = gesture_label


        self.gesture_raw_data = list()

        self.x = None
        self.y = None

        # self.x_train = None
        # self.y_train = None
        
        # self.x_test = None
        # self.y_test = None
    
    def get_accomplish_path_name(self,index):
        if not self.accomplish_path:
            self._get_file()
        return self.accomplish_path[index][0]
    
    def get_accomplish_path_total(self):
        if not self.accomplish_path:
            self._get_file()
        return len(self.accomplish_path)
    
    def _get_file(self):
        """
        
        """
        data_classes = set()
        accomplish_path = []
        path = self.path
        for subdir in listdir(path):
            for i in listdir(path + '/' + subdir):
                # 2_5-4_2021-07-27-06-36-43_jack.txt
                split_filename = i.split('_')
                total_gesture = split_filename[0]
                if total_gesture!= 0:
                    gesture_list = split_filename[1].split('-')
                    accomplish_path.append((path + '/' + subdir + '/' + i, gesture_list))
                    # data_classes.add(subdir)
                else:
                    accomplish_path.append((path + '/' + subdir + '/' + i, []))
                # 格式如：('train1&2_test2/testData2/0/02_2019-01-11-07-51-08_C.Y_Chen.txt', '0')
        self.accomplish_path = accomplish_path        
        


    def _get_raw_data_from_file(self,path):
        """
        
        """
        data = list()
        # print(path)
        with open(path,"r") as f:
            raw = f.readline()
            while raw:
                data.append( list(map(int,raw.split(",")[:-1])) ) 
                raw = f.readline()
        # print(data)
        # self.gesture_raw_data.append(data)
        return data


    def _find_gesture_label(self,data):
        """
        
        """
        label = list()
        for i,raw in enumerate(data):
            if raw == self.gesture_label:
                label.append(i)
                del data[i]
        
        if len(label)%2 != 0:
            # raise RuntimeError(f"Total gesture label {self.gesture_label}  is not 2")
            print(f"Total gesture label {self.gesture_label}  is not even")
            raise
        else:
            return label

    def _generate_ground_truth(self,data,label):
        """
        
        """
        ground_truth = [0]*len(data)
        if label is None:
            return 
        
        for i in range(0,len(label),2):
            ground_truth_len = label[i+1]-label[i]
        
            gaussian_ground_truth = Ground_truth(ground_truth_len,ground_truth_len/6)
            # gaussian_ground_truth.plot()
            # print(gaussian_ground_truth.truth)
            # print(len(data))
            # print(len(gaussian_ground_truth.truth))
            
            
            for j, truth in enumerate(gaussian_ground_truth.truth):
                ground_truth[j+label[i]] = truth

        # for i in range(label[0],len(gaussian_ground_truth.truth)):
        #     print(i-label[0])
        #     print(len(gaussian_ground_truth.truth))
        #     ground_truth[i] = gaussian_ground_truth.truth[i-label[0]]

        return ground_truth

    def generate_test_data(self,index):
        def _gen_grund_truth_class(data,label,gclass):
            grund_truth_class = [0]*len(data)
            if not label:
                return grund_truth_class
            # print(len(label)/2)
            # print(len(gclass))
            # print("*"*10)
            # str(len(label)/2)+ str(len(gclass))
            assert len(label)/2 == len(gclass),"123"
            for i,cls in zip(range(0,len(label),2),gclass):
                for j in range(label[i],label[i+1]):
                    grund_truth_class[j] = cls
            return grund_truth_class

        if not self.accomplish_path:
            self._get_file()
        raw_data = self._get_raw_data_from_file(self.accomplish_path[index][0])
        data_len = len(raw_data)
        if data_len < self.windows_size:
            print(f"file data {self.accomplish_path[index]} total line smaller than {self.windows_size}")

        label = self._find_gesture_label(raw_data)
        grund_truth = self._generate_ground_truth(raw_data,label)

        # data_classes = self.data_classes
        class_name = self.accomplish_path[index][1]
        # print(label)
        # print(class_name)
        grund_truth_class_list = _gen_grund_truth_class(raw_data,label,class_name)
        ret = {
            "raw_data":raw_data,
            "label":label,
            "ground_truth":grund_truth,
            "class_name":class_name,
            "ground_truth_class":grund_truth_class_list
        }

        return ret

    def generate_train_data(self):
        """

        """
        # get file path
        self._get_file()

        x = list()
        x_label = list()
        x_path = list()
        y_hm = list()
        y_wh = list()
        data_classes = self.data_classes
        data_classes_total = self.data_classes_total

        test_count = 0

        # for all file
        for j,p in enumerate(self.accomplish_path):
            # get file data
            raw_data = self._get_raw_data_from_file(p[0])
            raw_data_class = p[1]
            
            # muti class 
            data_classes_index = data_classes.index(raw_data_class)
            
            # single class
            # data_classes_index = 0
            
            # print(raw_data_class,data_classes_index)
            print(p)
            label = self._find_gesture_label(raw_data)
            grund_truth = self._generate_ground_truth(raw_data,label)
            # print(raw_data)
            # print(label)
            # print(grund_truth)

            # split data by slide windows 
            data_len = len(raw_data)
            if data_len < self.windows_size:
                print(f"file data {p} total line smaller than {self.windows_size}")
                continue
                # raise RuntimeError(f"file data {p} total line smaller than {self.windows_size}")
            if label:
                label_len = (label[1]-label[0]) //2
                label_mid_index = (label[1]+label[0]) //2
                
                assert label_len > 0

            for i in range(data_len-self.windows_size+1):
                split_data = raw_data[i:i+self.windows_size]
                split_ground_truth = grund_truth[i:i+self.windows_size]
                if max(split_ground_truth) != 1:
                    split_ground_truth = [0]*self.windows_size
                
                # x.append(normalize(split_data))
                x.append(np.array(split_data)/360)
                x_label.append(raw_data_class)
                x_path.append(p)
                # y.append({'hm': np.array(split_ground_truth) ,'wh':label_len  })
                
                all_classes_y = np.zeros((data_classes_total,self.windows_size), dtype=np.float32)
                all_classes_y[data_classes_index] = np.array(split_ground_truth)

                wh = np.zeros((self.windows_size), dtype=np.float32)
                if label:
                    if abs(label_mid_index-i) >= self.windows_size:
                        # print(p)
                        # print(label_mid_index)
                        pass
                    else:
                        # print(p)
                        # print(label_mid_index,i)
                        wh[label_mid_index-i] = label_len
          

                # if test_count==2007:
                #     print(p)
                #     print(data_classes_index)
                #     print(all_classes_y)
                #     print(all_classes_y.T.tolist())
                #     print(wh)
                #     input()
                test_count += 1
                y_hm.append(all_classes_y.T)
                y_wh.append(np.reshape(wh,(len(wh),1)))


            

                
            
        # format raw data to training data
        
        self.x = {"x":np.array(x),"path":x_path,"label":x_label}
        self.y = {"hm":np.array(y_hm),"wh":np.array(y_wh)}



    

    def plot(self, index,save_path):
        
        """
        
        """

        print(self.x['path'][index])
        data_classes = self.data_classes

        tottal_data_classes = len(self.data_classes)+1
        fig, (ax1, ax2) = plt.subplots(2,1)
        ax1.set_title(f"class type {self.x['label'][index]}")
        ax1.plot(self.x['x'][index])
        
        print(data_classes)
        # print(self.y['hm'][index].tolist())
        for i,hm in enumerate(self.y['hm'][index].T):
            # print(i)
            ax2.plot(hm,label=data_classes[i])
            # print(data_classes[i])
            # print(hm)
            mid,w =  np.argmax(hm),np.max(self.y['wh'][index])
            if mid != 0 and mid <= self.windows_size:
                print(f"class type {data_classes[i]}")
                # print(self.y['wh'][index])
                # print(self.y['wh'][index].shape)

                ax2.set_title(f"Ground truth w={w}")
                ax2.axvline(mid,linestyle='--')
                ax2.axvline(mid+w,linestyle='--')
                ax2.axvline(mid-w,linestyle='--')

                
                ax1.axvline(mid,linestyle='--')
                ax1.axvline(mid+w,linestyle='--')
                ax1.axvline(mid-w,linestyle='--')
                

        
        # print(self.y['hm'][index])
        # ax2.legend(bbox_to_anchor=(1.0, 1.0, 0.3, 0.2),loc = 'upper right')
        plt.tight_layout()
        # plt.show()
        plt.savefig('{}/{}.png'.format(save_path,f"class_type_{self.x['label'][index]}_{index}"))
    

    def analysis(self):
        p,n = [0]*len(self.data_classes), [0]*len(self.data_classes)
        for x,y in zip(self.x['path'],self.y['hm']):
            if np.max(y)==1:
                p[int(x[1])] += 1
            else:
                n[int(x[1])] += 1

        print(p)
        print(n)





if __name__ == "__main__":
    G = Gesture_Data(r"./testData")

    G.generate_test_data()



#Student: EhsanMasoudi.ir
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

class ClassScore:
    def __init__(self,klid_path,pasokhname_path):
        self.klid_path = klid_path
        self.pasokhname_path = pasokhname_path
    @property
    def klid(self):
        img = cv2.imread(self.klid_path)

        y1_start = 415
        y1_distance = 869
        y1_step = 68
        y_start = 32
        y_distance = 30
        y_step = 57

        lower = np.array([40,50,50])
        upper = np.array([62,255,255])
        True_answers = []

        for j in range(17):
            s1 = y1_start + j*y1_distance + j*y1_step
            e1 = y1_start + (j+1)*y1_distance + j*y1_step
            j_img = img[s1:e1,:]
            for k in range(10):
                if j == 16 and k == 5: break
                s1 = y_start + k*y_distance + k*y_step
                e1 = y_start + (k+1)*y_distance + k*y_step
                k_img = j_img[s1:e1,:]
                hsv_img = cv2.cvtColor(k_img, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv_img, lower, upper)
                sumation_list = [mask[:,290:340].sum(),mask[:,702:752].sum(),mask[:,1117:1166].sum(),mask[:,1523:1574].sum()]
                if sumation_list[0] > 27000:
                    True_answers.append(1)
                elif sumation_list[1] > 27000:
                    True_answers.append(2)
                elif sumation_list[2] > 27000:
                    True_answers.append(3)
                elif sumation_list[3] > 27000:
                    True_answers.append(4)
                else:
                    True_answers.append('-')
        True_answers[148] = 1
        return True_answers
    @property
    def pasokhnameh(self):
        img = cv2.imread(self.pasokhname_path)
        resized = cv2.resize(img, (1632,2390), interpolation = cv2.INTER_AREA)
        gray_img = cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)

        max_threshold = 130
        gray_img[gray_img >= max_threshold] = 255
        gray_img[gray_img < max_threshold] = 0

        kernel = np.ones((1,2), np.uint8)
        dilation = cv2.dilate(gray_img, kernel, iterations = 5)
        noise_removed_image = cv2.medianBlur(dilation,5)
        kernel = np.ones((2,2), np.uint8)
        closing = cv2.morphologyEx(noise_removed_image, cv2.MORPH_CLOSE, kernel)

        closing = closing[630:2171,55:1550]

        x_start = 90
        x_distance = 126
        x_step = 130
        y1_start = 10
        y1_distance = 273
        y1_step = 40
        y_start = 4
        y_distance = 18
        y_step = 10

        answers = []

        for i in range(5):#range: 6
            s = x_start + i*x_distance + i*x_step
            e = x_start + (i+1)*x_distance + i*x_step
            i_img = closing[:,s:e]
            for j in range(5):
                s1 = y1_start + j*y1_distance + j*y1_step
                e1 = y1_start + (j+1)*y1_distance + j*y1_step
                j_img = i_img[s1:e1,:]
                for k in range(10):
                    s2 = y_start + k*y_distance + k*y_step
                    e2 = y_start + (k+1)*y_distance + k*y_step
                    k_img = j_img[s2:e2,:]
                    if k_img[:,:30].sum() < 120000:
                        answers.append(1)
                    elif k_img[:,31:62].sum() < 112000:
                        answers.append(2)
                    elif k_img[:,62:93].sum() < 112000:
                        answers.append(3)
                    elif k_img[:,93:126].sum() < 112000:
                        answers.append(4)
                    else:
                        answers.append('-')
        return answers
    def score(self):
        Klid = self.klid
        Pasokhnameh = self.pasokhnameh
        trues = list(map(lambda x, y : x == y,Klid,Pasokhnameh[:len(Klid)]))
        return trues.count(True) / len(Klid) * 100
    def represention(self):
        Klid = self.klid
        Pasokh = self.pasokhnameh
    
        false = cv2.imread('pasokh/False.jpg')
        true = cv2.imread('pasokh/True.jpg')
        Ntrue = cv2.imread('pasokh/NTrue.jpg')
        head = cv2.imread('pasokh/head.jpg')
        kham = cv2.imread('pasokh/kham.jpg')
        head_height = head.shape[0]
        kham_height = kham.shape[0]
        height = head_height + kham.shape[0] * len(Klid)
        width = head.shape[1]
        pasokhbarg = np.zeros((height, width, 3), dtype=np.uint8)
        pasokhbarg[:head.shape[0],:head.shape[1]] = head
        x = [269,683,1090,1501]
        x_ = [319,733,1140,1551]
        for i in range(len(Klid)):
            current = np.copy(kham)
            if Klid[i] == Pasokh[i]:
                current[13:40,x[Pasokh[i]-1]:x_[Pasokh[i]-1]] = true
            elif Klid[i] != Pasokh[i] and Pasokh[i] != '-':
                current[13:40,x[Klid[i]-1]:x_[Klid[i]-1]] = Ntrue
                current[13:40,x[Pasokh[i]-1]:x_[Pasokh[i]-1]] = false
            else:
                current[13:40,x[Klid[i]-1]:x_[Klid[i]-1]] = Ntrue
            cv2.putText(current, str(i+1), (30,35), cv2.FONT_HERSHEY_COMPLEX, 1, (33,33,33), 2)
            pasokhbarg[(head_height+i*kham_height):(head_height+(i+1)*kham_height)
        ,:] = current
        file_name = self.pasokhname_path.split('/')[1].split('.')[0]
        cv2.imwrite('status/' + file_name + '.jpg',pasokhbarg)
        cv2.imshow(file_name,pasokhbarg)
        cv2.waitKey()
    def save_status(self):
        Klid = self.klid
        Pasokhnameh = self.pasokhnameh
        status_list = []
        for i in range(len(Klid)):
            if Klid[i] == Pasokhnameh[i]:
                status_list.append(True)
            elif Klid[i] != Pasokhnameh[i] and Pasokhnameh[i] != '-':
                status_list.append(False)
            else:
                status_list.append('-')
        
        file_name = self.pasokhname_path.split('/')[1].split('.')[0]
        df = pd.DataFrame(data=status_list,columns=[file_name],index=range(1,len(Klid)+1))
        df.to_csv('status/' + file_name + '.csv')
        return df
    @staticmethod
    def save_allstatus():
        images = os.listdir('ResponseLetter/')
        Klid = ClassScore('kild.png','').klid
        df_list = []
        for image in images:
            Pasokhnameh = ClassScore('kild.png','ResponseLetter/' + image).pasokhnameh
            status_list = []
            for i in range(len(Klid)):
                if Klid[i] == Pasokhnameh[i]:
                    status_list.append(True)
                elif Klid[i] != Pasokhnameh[i] and Pasokhnameh[i] != '-':
                    status_list.append(False)
                else:
                    status_list.append('-')
            df_list.append(status_list)
        df_list = np.array(df_list)
        df_list = pd.DataFrame(df_list.T,columns=images,index=range(1,len(Klid)+1))
        df_list.to_csv('allstatus.csv')
        return df_list
    @staticmethod
    def save_all():
        images = os.listdir('ResponseLetter/')
        Klid = ClassScore('kild.png','').klid
        Score_list_all = []
        for image in images:
            Pasokhnameh = ClassScore('kild.png','ResponseLetter/' + image).pasokhnameh
            trues = list(map(lambda x, y : x == y,Klid,Pasokhnameh[:len(Klid)]))
            Score_list_all.append(trues.count(True) / len(Klid) * 100)
        df = pd.DataFrame(Score_list_all,index=images,columns=['Score(%)'])
        df.to_csv('allScores.csv')
        return df



stu = ClassScore('kild.png','ResponseLetter/image0000013A.tif')
#print(stu.pasokhnameh) #This property got the student's answers from the answer sheet image
#print(stu.klid) #This property got the correct answers from the kild.png image
#stu.represention() #This method designs and show a beautiful report card of the status to guestions and also save the picture in the status directory
#print(stu.score()) #This method calculates the percentage of correct answers
#print(stu.save_status()) #This method shows the student's status of questions and save them into a csv file in the "status" directory
#print(ClassScore.save_allstatus()) #This method shows all students' status of all questions and they will be saved into a csv file('allstatus.csv')
#print(ClassScore.save_all()) #This method calculates Scores of all students and save them into a csv file('allScores.csv')


import serial  # 藍芽函式庫 pip install pyserial
import struct  # 二進制字串打包處理
import numpy
import os  # 目錄生成與檔案控制 
import time  # 抓取日期與時間
import msvcrt  # 按鍵監聽 

from serial_usb import serialUSB

###################### ###################### 
###      設定檔案路徑、使用者以及手勢       ###
############################################ 
  
# PATH = 'testData'  # 接收檔案儲存路徑 
PATH = 'trainData'  
USER = "chieh"  # 資料蒐集者 
DATE = time.strftime('%Y-%m-%d',   time.localtime(time.time()))  # 當天日期
GESTURE = "6"  # 手勢動作
DATASET = "TR" 




SENSOR_NUM = 3

# 空白鍵按鍵監聽 
def hit_key():
    if (msvcrt.kbhit()):  # 若偵測到鍵盤事件
        if (ord(msvcrt.getch()) == 32):  # 若鍵盤事件為 空白鍵 = 32
            return True
    return False

if __name__ == '__main__':

    # folder = f"{PATH}/{USER}/{DATE}-{DATASET}/{GESTURE}"
    folder = f"{PATH}/{GESTURE}"
    if not os.path.exists(folder):
        os.makedirs(folder) # 產生檔案儲存路徑


    
    # split_data = numpy.zeros((SENSOR_NUM,2), dtype=int) 
    # full_data = numpy.zeros(SENSOR_NUM,dtype=int)
    
    STOP_FLAG = False

    
    portName = 'COM5'     # for windows users
    baudRate = 115200
    maxPlotLength = 100
    dataNumBytes = 2      # number of bytes of 1 data point
    dataInNum = 3
    serialData = serialUSB(portName, baudRate, maxPlotLength, dataNumBytes,dataInNum)   # initializes all required variables

    

    try:
        data_total = 0
        while (1):
            data_total += 1
            # 等待開始
            # while not hit_key(): 
            #     SERIAL.flushInput()   # 清空 Comport's receive Buffer 
            #     print('waiting')
            print('waiting')
            while(1):
                c = msvcrt.getch()
                if(c==b' '):
                    break
                elif(c==b'\x1b'): # ESC
                    STOP_FLAG = True
                    break
            
            if STOP_FLAG:
                print("program stop")
                break

                
            print(f"Start recording data {data_total}")

            deTime = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))  # deTime = 當前詳細時間
            with open(f"{folder}/{GESTURE}_{deTime}_{USER}_{DATASET}.txt","w") as sensor_File:
                # 開始紀錄回傳的感測值
                serialData.serialConnection.reset_input_buffer()
                while not hit_key():
                    check = serialData.serialConnection.read().decode("ISO-8859-1") # Bluetooth 接收與解譯
                    # print(check)
                    if check == 'S':
                        for i in range(dataInNum):
                            raw = serialData.serialConnection.read(2)
                            data = int.from_bytes(raw, byteorder='little', signed=True) * -1  
                            sensor_File.write("{0:4d},".format(data))
                        sensor_File.write("\n")  

            print("END recording data")
    finally:
        serialData.serialConnection.close()
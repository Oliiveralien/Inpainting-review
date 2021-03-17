import os
import os.path
import shutil
import time, datetime

def get_file(num):
    file = open("places365_val.txt","r")
    lines = file.read().split('\n')
    ress = [line.split(' ')[:2] for line in lines]
    ress = [res[0] for res in ress if res[-1] == str(num)]
    return ress  

def moveFileto(sourceDir,  targetDir): 
    shutil.copy(sourceDir,  targetDir)
    
def move(num):
    file_names = get_file(81)
    rootpath = "dataset\\val_256\\"
    for file_name in file_names:
        path = rootpath + file_name
        moveFileto(path, "datasets\\val")
    print("done")
    
move(81)
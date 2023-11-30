## steps for implementing the background model
import cv2
import math
from matplotlib import pyplot as plt

no_of_frames = 6
imgarr = []             #list of frames
for no in range(1,no_of_frames+1):
        scene = cv2.imread('office (%d).jpg' % no,0)
        imgarr.append(scene)

res_backmod = []
print imgarr[0].shape

'''''''moving block wise to each frame '''''''

row,col = (imgarr[0]).shape
for oi in range((row/10)*2-1):                        ##applying a block of 10X10 to each frame and then moving to next block 
        for oj in range((col/10)*2-1):                 
            model_hist_block = []           ##list of hist of particular block
            wt = []                         #list of weights of corres hist's
            for a in range(no_of_frames):
                block = imgarr[a][(oi*5):((oi*5)+10),(oj*5):((oj*5)+10)]
                block_lbp = block.copy()
                for i in range(10):
                    for j in range(10):
                        su = 0
                        p = -1
                        for m in range(-1,2):
                            for n in range(-1,2):
                                if((abs(m)+abs(n))>0):
                                        p = p+1
                                        if((i+m>=0 and j+n>=0)and(i+m<10 and j+n<10)):
                                                if(block[i+m][j+n]>=block[i][j]):
                                                        su = su + pow(2,p)
                        block_lbp[i,j] = su                
                block_hist = cv2.calcHist([block_lbp],[0],None,[256],[0,256])           #receiving block_hist from frame 
                match = 0
                for m in range(len(model_hist_block)):
                    #comp and update ''model_hist_block''
                    su = 0
                    for n in range(256):
                        su = su + min(model_hist_block[m][n],block_hist[n])
                    if int(su) >= 45:
                        match = 1
                        #update the matched histogram bins
                        histcpy = (model_hist_block[m])
                        for n in range(256):
                            histcpy[n] = (1*block_hist[n] + (100-1)*model_hist_block[m][n])/100    
                        #histcpy1 = [n*100/sum(histcpy) for n in histcpy]    
                        model_hist_block[m] = histcpy               #updating the matched histgoram
                        #update the weights
                        wt[m]+=1
                if(match == 0):
                        if(len(model_hist_block)==3):
                            #find the lowest wt hist
                            low_wt = wt.index(min(wt)) 
                            model_hist_block[low_wt] = block_hist
                            wt[low_wt] = 1
                            #wt normalization
                        else:
                            wt.append(1)
                            model_hist_block.append(block_hist)
                            
            #take the most weighted hist
            model = model_hist_block[wt.index(max(wt))]
            res_backmod.append(model)

print 'done_background_model'

det_arr = []
det_arr_g = []
for i in range(7,14):
        img = cv2.imread('office (%d).jpg'% i)
        img_g = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        det_arr.append(img)
        det_arr_g.append(img_g)
#scene1 = cv2.imread('office (8).jpg',0)
mrow,mcol = det_arr_g[0].shape
for x in range(len(det_arr)):
        imgres = det_arr[x].copy()
        arr = [0 for i in range((mrow/5)*(mcol/5))]
        no_times = [0 for i in range((mrow/5)*(mcol/5))]
        for mi in range((mrow/10)*2-1):                        ##applying a block of 10X10 to each frame and then moving to next block 
                for mj in range((mcol/10)*2-1):
                    mblock = det_arr_g[x][(mi*5):(mi*5)+10,(mj*5):(mj*5)+10]
                    mblock_lbp = mblock.copy()
                    for i in range(10):
                        for j in range(10):
                            su = 0
                            p = -1
                            for m in range(-1,2):
                                for n in range(-1,2):
                                    if((abs(m)+abs(n))>0): 
                                        p=p+1
                                        if((i+m>=0 and j+n>=0)and( i+m<10 and j+n<10)):
                                            if(mblock[i+m][j+n]>=mblock[i][j]):
                                                su = su + pow(2,p)
                            mblock_lbp[i,j] = su
                    mblock_hist = cv2.calcHist([mblock_lbp],[0],None,[256],[0,256])

                    count = (mi)*((mcol/5)-1) + mj
                    su = 0
                    for n in range(256):
                        su = su + min(res_backmod[count][n],mblock_hist[n])
                    x_val = (mi*mcol/5)+(mj)
                    #print x_val,x_val+mcol/5
                    arr[x_val] = arr[x_val]+su
                    no_times[x_val]+=1
                    arr[x_val+1] = arr[x_val+1]+su
                    no_times[x_val+1]+=1
                    arr[x_val+mcol/5] = arr[x_val+mcol/5]+su
                    no_times[x_val+mcol/5]+=1
                    arr[x_val+mcol/5+1] = arr[x_val+mcol/5+1]+su
                    no_times[x_val+mcol/5+1]+=1

        for i in range(len(arr)):
               arr[i] = arr[i]/no_times[i]
        for i in range(len(arr)):
               x_val = i/(mcol/5)
               y_val = i%(mcol/5)
               if arr[i]>=45:
                   imgres[(x_val*5):(x_val*5)+5,(y_val*5):(y_val*5)+5] = 0
        cv2.imshow('forgroundimage',imgres)
        cv2.imwrite('fore_.45Td(%d).jpg' % (x+7),imgres)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

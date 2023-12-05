###################""SILTP for complete video""##########################
import cv2
import math
import numpy as np

imgarr = []
no_frame = 12
size_Bblock = 10
tow = 0.05
alpha = 0.005
#pb_Bblock = [0 for i in range(((row/size_Bblock)*2-1)*((col/size_Bblock)*2-1))]
neta = 1.0  #in float

Ts = 0.50
imgarr_c= []
Wc = 1.0/50

for i in range(1,no_frame+1):
    img = cv2.imread('office (%d).jpg' % i)
    img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgarr.append(img_g)
    imgarr_c.append(img)

row,col = imgarr[0].shape
print row,col
'''imgcpy = (imgarr[no_frame-1]).copy()'''
Bblockarr_siltphist = [[ 1.0/(pow(3,8)) for i in range(pow(3,8))] for j in range(((row/10)*2-1)*((col/10)*2-1))]

blue_back_model = np.zeros((row,col))
green_back_model = np.zeros((row,col))
red_back_model = np.zeros((row,col))
for i in range(row):
    for j in range(col):
        blue_back_model[i,j] = int(imgarr_c[0][:,:,0][i][j])
        green_back_model[i,j] = int(imgarr_c[0][:,:,1][i][j])
        red_back_model[i,j] = int(imgarr_c[0][:,:,2][i][j])

for x in range(no_frame):
    curr_frame = (imgarr[x]).copy()
    curfram_c = imgarr_c[x].copy()
    blue_curfram = curfram_c[:,:,0].copy()
    green_curfram = curfram_c[:,:,1].copy()
    red_curfram = curfram_c[:,:,2].copy()
    pb_Sblock = [0 for i in range((row*2/10)*(col*2/10))]
    no_times = [0 for i in range((row*2/10)*(col*2/10))]
    for i in range((row/10)*2-1):
        for j in range((col/10)*2-1):
            aise_Bblockhist = [0 for r in range(pow(3,8))]
            #hist_Bblock = [0 for i in range(pow(3,8))]      #where 8 neighbors are there
            curr_Bblockmodel = Bblockarr_siltphist[i*((col/10)*2-1)+j]
            B_block = (curr_frame[(i*5):(i*5)+10,j*5:(j*5)+10]).copy()
            for k in range(10):
                for l in range(10):
                    su = 0
                    p = -1
                    for m in range(-1,2):
                        for n in range(-1,2):
                            if(abs(m)+abs(n)>0):
                                p +=1
                                if( 0<=(k+m)<10 and 0<=(l+n)<10):
                                    if(B_block[k+m][l+n]>(1+tow)*B_block[k][l]):
                                        su = su + pow(3,p)
                                    elif(B_block[k+m][l+n]<(1-tow)*B_block[k][l]):
                                        su = su + 2*pow(3,p)
                    aise_Bblockhist[su] = aise_Bblockhist[su] + 1.0/(10*10)   
                    '''#updation of big block model having only one model for each big block
                    if(x!=(no_frame-1)):
                        curr_Bblockmodel[su] = (1-alpha)*curr_Bblockmodel[su] + alpha*(1.0/(10*10))
                    #if last frame: then object detection    
                    else:
                        hist_finalframeblocks[su] = hist_finalframeblocks[su] + 1.0/(size_Bblock*size_Bblock)'''
            
            if(x<6):
                curr_Bblockmodel_cpy = [0 for r in range(pow(3,8))]
                for q in range(pow(3,8)):
                    curr_Bblockmodel_cpy[q] = (1-0.005)*curr_Bblockmodel[q] + 0.005*aise_Bblockhist[q]              #alpha = 0.005
                Bblockarr_siltphist[i*((col/10)*2-1)+j] = curr_Bblockmodel_cpy

            else:
                pbBblock_curr = 0.0
                for y in range(pow(3,8)):
                    if(curr_Bblockmodel[y]>=(1.0/pow(3,8))):     #neta = 1.0
                        #pb_Bblock[i*(col*2/size_Bblock-1)+j] += hist_finalframeblocks[y]
                        pbBblock_curr = pbBblock_curr + aise_Bblockhist[y]
                #print pbBblock_curr
                #Also splitting out the probabilities into small blocks:
                index_val = i*(col/10)*2+j
                #print index_val, i, j
                pb_Sblock[index_val] = pb_Sblock[index_val] + pbBblock_curr
                no_times[index_val] += 1
                pb_Sblock[index_val+1] = pb_Sblock[index_val+1] + pbBblock_curr
                no_times[index_val+1] += 1
                pb_Sblock[index_val+col*2/10] = pb_Sblock[index_val+col*2/10] + pbBblock_curr
                no_times[index_val+col*2/10] += 1
                pb_Sblock[index_val+col*2/10+1] = pb_Sblock[index_val+col*2/10+1] + pbBblock_curr
                no_times[index_val+col*2/10+1] += 1  
    if(x>=6):
        for i2 in range(len(pb_Sblock)):
            pb_Sblock[i2] = pb_Sblock[i2]/no_times[i2]
            #print pb_Sblock[i],'pbsmallblock', i, no_times[i]
    #############################color_model#########################


    if(0<x<6):
        for i1 in range(row):
            for j1 in range(col):
                blue_back_model[i1,j1] = (1-Wc)*(blue_back_model[i1][j1]) + Wc*(blue_curfram[i1][j1])
                green_back_model[i1,j1] = (1-Wc)*(green_back_model[i1][j1]) + Wc*(green_curfram[i1][j1])
                red_back_model[i1,j1] = (1-Wc)*(red_back_model[i1][j1]) + Wc*(red_curfram[i1][j1])
    elif(x>=6):
        test_img = imgarr_c[x].copy()
        cv2.imshow('img',test_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        for i3 in range(row/5):
            for j3 in range(col/5):
                Db=0
                Dg=0
                Dr=0
                D = 0
                sblock_curfram_b = blue_curfram[i3*5:i3*5+5,j3*5:j3*5+5].copy()
                sblock_curfram_g = green_curfram[i3*5:i3*5+5,j3*5:j3*5+5].copy()
                sblock_curfram_r = red_curfram[i3*5:i3*5+5,j3*5:j3*5+5].copy()
                sblock_backmodel_b = blue_back_model[i3*5:i3*5+5,j3*5:j3*5+5].copy()
                sblock_backmodel_g = green_back_model[i3*5:i3*5+5,j3*5:j3*5+5].copy()
                sblock_backmodel_r = red_back_model[i3*5:i3*5+5,j3*5:j3*5+5].copy()
                for k in range(5):
                    for l in range(5):
                        Db = Db + (int(sblock_backmodel_b[k][l]) - int(sblock_curfram_b[k][l]))
                        #print Db,'Db'
                        Dg = Dg + (int(sblock_backmodel_g[k][l]) - int(sblock_curfram_g[k][l]))
                        Dr = Dr + (int(sblock_backmodel_r[k][l]) - int(sblock_curfram_r[k][l]))
                #print Db,Dg,Dr
                D = (pow((Db/25),2.0)+pow((Dg/25),2.0)+pow((Dr/25),2.0))/(255*255*3)
                #print D
                #####condition for texture + color
                ##if(D<0.05 and pb_Sblock[i3*(col/5) + j3]>Ts):
                #    test_img[i3*5:i3*5+5,j3*5:j3*5+5] = 0
                #####condition for high quality equ
                #if((1-pb_Sblock[i3*(col/5) + j3])*D<(1-Ts)*0.05/9):
                #    test_img[i3*5:i3*5+5,j3*5:j3*5+5] = 0
                #####combined all 3 eqs
                if(D<0.05 and pb_Sblock[i3*(col/5) + j3]>Ts and (1-pb_Sblock[i3*(col/5) + j3])*D<(1-Ts)*0.05/9):
                    test_img[i3*5:i3*5+5,j3*5:j3*5+5] = 0
                #if (pb_Sblock[i*(col/5) + j]>Ts):
                #    test_img[i*5:i*5+5,j*5:j*5+5] = 0
                #if ((1-pb_Sblock[i*(col/5) + j])*(1-D) > (1.0/9)*(1-Ts)*0.2):
                #    test_img[i*5:i*5+5,j*5:j*5+5] = 0
        cv2.imshow('tex+col+3rd_eq',test_img)
        cv2.imwrite('tex+col+3rd_eq_tow0.05_%d.jpg' % x,test_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print 'done %d frame' % x

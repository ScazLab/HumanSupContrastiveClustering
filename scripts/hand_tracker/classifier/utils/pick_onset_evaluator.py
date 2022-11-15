import numpy as np
from numpy.lib.function_base import append


def compare_array(pick_gts, pick_preds, find, idx, pred_count_list):
    # gt_seg = pick_gts[idx, :]
    # pred_seg = pick_preds[find, :]
    count = 1
    return_idx = 0
    prev = -1
    # for idx, (gt, pred) in enumerate(zip(gt_seg, pred_seg)):
    while idx < len(pick_gts) and find < len(pick_preds):
         
        pred_count_list[pick_preds[find]] = count
        if pick_gts[idx] != pick_preds[find]:

            break
        prev = pick_gts[idx]
        count+=1
        idx+=1
        find+=1
        # print(count, find, idx)
    return count, find, idx,  pred_count_list, prev
    


pred_gt_array = np.load('preds_20.npy', allow_pickle=True)

index_array = []
window = []
count = 0
correct = 0
gt_count = 0
total_gt_count = 0

preds = pred_gt_array[:, 1]
gts = pred_gt_array[:, 2]
print(len(gts), len(preds))

pick_preds = np.where(preds==1)[0]
pick_gts = np.where(gts==1)[0]
print(pick_gts)
print(pick_preds)
tp = 0


pred_count_list = [0]*len(preds)
visited_list = []
k = 3
i = 0
while i < len(pick_gts):
    #print(gt)
    gt = pick_gts[i]
    #print(i)
    find = np.where(pick_preds==gt)[0]

    #print(i, gt, find)
    if len(find)!=0:
        visited_list.append(pick_preds[find])
        find = find[0]
        #print(i,find)
        count, find, i,  pred_count_list, prev = compare_array(pick_gts, pick_preds, find, i, pred_count_list)
        #print(count,find, i)
        

        
        if count >= 1:
            tp += 1
            print(pick_gts[i-1])
            if pick_gts[i-1] != prev:
                continue
            start = i
            end = start + 1

            while pick_gts[end-1] == pick_gts[start]:
                start += 1
                end += 1
                if end >= len(pick_gts):
                    break
            i = end
        else:
            i = i + 1
    else:            
         i = i+1
    
total_gt = 0
i = 0
print(visited_list)

gt_count_list = [0]*len(preds)


while i < len(pick_gts):
    start = i
    end = start + 1
    count = 1

    
    gt_count_list[pick_gts[i]] = count 

    while pick_gts[end] == pick_gts[start]+1:
        
        count = count + 1
        #print(pick_gts[start], pick_gts[end], count)
        gt_count_list[pick_gts[end]] = count 
        
        start += 1
        end += 1
        if end >= len(pick_gts):
            break
    #print(count)
    if count>=1:
         total_gt = total_gt + 1
    i = end
    #print(end)




#print(len(preds), len(gt_count_list), len(pred_count_list))
gt_count_list = np.array(gt_count_list)
pred_count_list = np.array(pred_count_list)
count_list = np.stack((pred_count_list, gt_count_list))
np.save('pred_gt_counts.npy', count_list)
print(tp, total_gt, (tp/total_gt)*100)

'''
for index, i in enumerate(pred_gt_array):
    name = i[0][0]
    pred = i[1]
    gt = i[2]
    
    if pred == 1:
        print(name, pred, gt)
        count +=1
        window.append(index)
       
    else:
        if count >= 3:
            index_array.append(window[0])
            print(window)
            window = []
            count = 0
        print("-----")

    if gt==1:
        if pred!=1:
            print(name, pred, gt, "missed pick")
        gt_count += 1
    else:
        if gt_count >= 3:
            total_gt_count +=1
            gt_count = 0
        

            
onset_array = pred_gt_array[index_array]
print(onset_array)
for i in onset_array:
    name = i[0][0]
    pred = i[1]
    gt = i[2]
    if pred == gt:
        # print(name)
        correct +=1
    # correct += (pred == gt)
print(correct, total_gt_count)
acc = (correct/total_gt_count) * 100
print(acc)
'''


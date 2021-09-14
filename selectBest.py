# this script is used to select the best iter of network
import pandas as pd
import numpy as np
import os
import json

# BASE_PATH = "total_eval_result"
BASE_PATHS = ["save_result_dataset1","save_result_dataset2", "save_result_dataset3"]
BASE_PATH = "save_result_total"
netnames = os.listdir(BASE_PATH)

def getBestIter(base_path):
    f = open("bestIter.txt", "w")
    best_iter = {}
    for netname in netnames:
        results = os.listdir(os.path.join(base_path, netname))
        best_iter[netname] = (0, 0)
        for result in results:
            data = pd.read_csv(os.path.join(base_path, netname, result))
            fwiou = data["w_miou"].mean()
            iter = int(result.split("_")[-1].split(".")[0])
            if fwiou > best_iter[netname][1]:
                best_iter[netname] = (iter, fwiou)
    d_order=sorted(best_iter.items(),key=lambda x:x[1][1],reverse=False)
    for d in d_order:
        print('"',d[0],'"', sep="", end=",")
    print(json.dumps(best_iter), file=f)
    print(best_iter)

# merge three datasets results
def mergeTable():
    for netname in netnames:
        results = os.listdir(os.path.join(BASE_PATH, netname))
        for result in results:
            merge_data = None
            for base_path in BASE_PATHS:
                data = pd.read_csv(os.path.join(base_path, netname, result))
                if merge_data is None:
                    merge_data = data
                else:
                    merge_data = pd.concat([merge_data, data], axis=0, ignore_index=True)
            merge_data.to_csv(os.path.join("total_eval_result", netname, result))
# mergeTable()
getBestIter("total_eval_result")
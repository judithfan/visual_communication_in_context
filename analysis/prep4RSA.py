import os
import numpy as np

if __name__ == "__main__":

    layers = ['pool1','conv42','fc6']
    split_type = 'balancedavg'
    splits = map(str,np.arange(1,6))

    for layer in layers:
        for split_num in splits:
            cmd_string = "python generate_bdainput.py --adaptor_type multimodal_{} --split_type {}{}".format(layer,split_type,split_num)
            print '{}'.format(cmd_string)
            os.system(cmd_string)

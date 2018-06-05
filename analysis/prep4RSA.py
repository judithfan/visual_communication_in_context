import os
import numpy as np

if __name__ == "__main__":

    layers = ['pool1','conv42','fc6']
    # layers = ['fc6']
    split_type = 'balancedavg'
    # splits = ['16','32','64','128','256','512']
    splits = map(str,np.arange(1,6))

    ## for the adapted encoder
    for layer in layers:
        for split_num in splits:
            cmd_string = "python generate_bdainput.py --adaptor_type multimodal_{} --split_type {}{}".format(layer,split_type,split_num)
            print '{}'.format(cmd_string)
            os.system(cmd_string)

            ## make bdaOutput subdirectories to save out to
            bda_out_path = '../models/bdaOutput/multimodal_{}_{}{}'.format(layer,split_type,split_num)
            if not os.path.exists(bda_out_path):
                os.makedirs(bda_out_path)
                os.makedirs(os.path.join(bda_out_path,'raw'))

    ## also for humans
    for split_num in splits:
        cmd_string = "python generate_bdainput.py --adaptor_type human --split_type {}{} --gen_similarity True --gen_centroid True".format(split_type,split_num)
        print '{}'.format(cmd_string)
        os.system(cmd_string)

        ## make bdaOutput subdirectories to save out to
        bda_out_path = '../models/bdaOutput/human_{}{}'.format(split_type,split_num)
        if not os.path.exists(bda_out_path):
            os.makedirs(bda_out_path)
            os.makedirs(os.path.join(bda_out_path,'raw'))

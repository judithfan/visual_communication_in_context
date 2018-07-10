from __future__ import division
import os
import thread
import numpy as np

### python RSA.py --wppl BDA --perception human multimodal_fc6 multimodal_conv42 multimodal_pool1 --pragmatics combined S1 S0 --production cost nocost --split_type balancedavg1 balancedavg2 balancedavg3 balancedavg4 balancedavg5
### python RSA.py --wppl BDA --perception multimodal_conv42 --pragmatics S0 --production nocost --split_type balancedavg1 balancedavg2 balancedavg3 balancedavg4 balancedavg5
### python RSA.py --wppl evaluate --perception human --pragmatics combined --production cost --split_type balancedavg5
### python RSA.py --wppl evaluate --perception multimodal_fc6 --pragmatics combined --production cost --split_type balancedavg3
### python RSA.py --wppl evaluate --perception multimodal_conv42 --pragmatics combined --production cost --split_type balancedavg1 balancedavg2 balancedavg3 balancedavg4 balancedavg5
### python RSA.py --wppl evaluate --perception multimodal_fc6 --pragmatics S0 --production cost --split_type balancedavg1 balancedavg2 balancedavg3 balancedavg4 balancedavg5
### python RSA.py --wppl evaluate --perception multimodal_fc6 --pragmatics combined --production nocost --split_type balancedavg1 balancedavg2 balancedavg3 balancedavg4 balancedavg5
### python RSA.py --wppl BDA-enumerate --sim_scaling_lb 1 --sim_scaling_ub 200 --step_size 2 --split_type balancedavg1

def run_bda(perception, pragmatics, production, split_type):
    cmd_string = 'webppl BDA.wppl --require ./refModule/ -- --perception {} --pragmatics {} --production {} --splitType {}'.format(perception, pragmatics, production, split_type)
    print 'Running: {}'.format(cmd_string)
    thread.start_new_thread(os.system,(cmd_string,))

def run_bda_enumerate(simScaling, splitType):
    cmd_string = 'webppl BDA-enumerate.wppl --require ./refModule/ -- --simScaling {} --splitType {}'.format(simScaling, splitType)
    print 'Running: {}'.format(cmd_string)
    thread.start_new_thread(os.system,(cmd_string,))

def run_evaluate(perception, pragmatics, production, split_type):
    cmd_string = 'webppl evaluate.wppl --require ./refModule/ -- --paramSetting {}_{}_{} --adaptorType {} --splitType {}'.format(perception, pragmatics, production, perception, split_type)
    print 'Running: {}'.format(cmd_string)
    thread.start_new_thread(os.system,(cmd_string,))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--wppl', type=str, help='options: BDA | BDA-enumerate | evaluate', default='BDA')
    parser.add_argument('--perception', nargs='+', type=str, \
                        help='option: options: human| multimodal_conv42 | multimodal_pool1 | multimodal_fc6',\
                        default = 'multimodal_conv42')
    parser.add_argument('--pragmatics', nargs='+', type=str, \
                        help='option: combined | S1 | S0',\
                        default = 'combined')
    parser.add_argument('--production', nargs='+', type=str, \
                        help='option: cost | nocost',\
                        default = 'cost')
    parser.add_argument('--split_type', nargs='+', type=str, \
                        help='option: splitbyobject | alldata | balancedavg',\
                        default = 'balancedavg')
    parser.add_argument('--sim_scaling_lb', type=float, \
                        help='for BDA-enumerate only: this is the LOWER bound for the simScaling param. \
                              We will sweep through values from (sim_scaling_lb, sim_scaling_ub) in step_size sized steps',\
                        default = 1.0) 
    parser.add_argument('--sim_scaling_ub', type=float, \
                        help='for BDA-enumerate only: this is the UPPER bound for the simScaling param. \
                              We will sweep through values from (sim_scaling_lb, sim_scaling_ub) in step_size sized steps',\
                        default = 200.0) 
    parser.add_argument('--step_size', type=float, \
                        help='for BDA-enumerate only: this is the step size we will use to march through \
                              the simScaling range',\
                        default = 2.0)   

    args = parser.parse_args()

    perception = args.perception
    production = args.production
    pragmatics = args.pragmatics
    split_type = args.split_type
    lb = args.sim_scaling_lb
    ub = args.sim_scaling_ub
    step_size = args.step_size

    assert args.wppl in ['BDA','evaluate', 'BDA-enumerate']

    ## first run BDA-enumerate.wppl
    if 'BDA-enumerate' in args.wppl:
        ss_range = np.arange(lb,ub,step_size)
        for i,ss in enumerate(ss_range):        
            for split in split_type:
                run_bda_enumerate(ss,split)

    ## first run BDA.wppl
    elif 'BDA' in args.wppl:
        for perc in perception:
            for prag in pragmatics:
                for prod in production:
                    for split in split_type:
                        run_bda(perc,prag,prod,split)

    elif 'evaluate' in args.wppl:
        ## then on output, run evaluate.wppl
        for perc in perception:
            print perc
            for prag in pragmatics:
                for prod in production:
                    for split in split_type:
                        run_evaluate(perc,prag,prod,split)

    else:
        print '{} wppl command not recognized'.format(args.wppl)

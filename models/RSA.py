from __future__ import division
import os
import thread
import subprocess
import numpy as np
import analysis_helpers as h

### python RSA.py --wppl BDA --perception human multimodal_fc6 multimodal_conv42 multimodal_pool1 --pragmatics combined S0 --production cost nocost --split_type balancedavg1 balancedavg2 balancedavg3 balancedavg4 balancedavg5
### python RSA.py --wppl BDA --perception human multimodal_fc6 multimodal_conv42 multimodal_pool1 --pragmatics combined S0 --production cost nocost --split_type balancedavg1 
### python RSA.py --wppl BDA --perception multimodal_conv42 --pragmatics S0 --production nocost --split_type balancedavg1 balancedavg2 balancedavg3 balancedavg4 balancedavg5
### python RSA.py --wppl evaluate --perception human --pragmatics combined --production cost --split_type balancedavg1
### python RSA.py --wppl evaluate --perception multimodal_fc6 --pragmatics combined --production cost --split_type balancedavg3
### python RSA.py --wppl evaluate --perception multimodal_conv42 --pragmatics combined --production cost --split_type balancedavg1 balancedavg2 balancedavg3 balancedavg4 balancedavg5
### python RSA.py --wppl evaluate --perception multimodal_fc6 --pragmatics S0 --production cost --split_type balancedavg1 balancedavg2 balancedavg3 balancedavg4 balancedavg5
### python RSA.py --wppl evaluate --perception multimodal_fc6 --pragmatics combined --production nocost --split_type balancedavg1 balancedavg2 balancedavg3 balancedavg4 balancedavg5
### python RSA.py --wppl BDA-enumerate --sim_scaling_lb 1 --sim_scaling_ub 200 --step_size 2 --split_type balancedavg5
### python RSA.py --wppl AIS --perception multimodal_fc6 --production cost --pragmatics combined --num_ais_samples 2 --split_type balancedavg1
### python RSA.py --wppl flatten

def run_bda(perception, pragmatics, production, split_type):
    if not os.path.exists('./bdaOutput'):
        os.makedirs('./bdaOutput')  
    if not os.path.exists('./bdaOutput/{}_{}'.format(perception,split_type)):
        os.makedirs('./bdaOutput/{}_{}'.format(perception,split_type))
        os.makedirs('./bdaOutput/{}_{}/raw'.format(perception,split_type))
    # check to make sure we do not already have output
    if not os.path.exists('./bdaOutput/{}_{}/raw/{}_{}_{}_{}Params.csv'.format(perception,\
                                                                                split_type,\
                                                                                perception,\
                                                                                pragmatics,\
                                                                                production,\
                                                                                split_type)):  
        #sample: models/bdaOutput/human_balancedavg1/raw/human_combined_cost_balancedavg1Params.csv
        cmd_string = 'webppl BDA.wppl --require ./refModule/ -- --perception {} --pragmatics {} --production {} --splitType {}'.format(perception, pragmatics, production, split_type)
        print 'Running: {}'.format(cmd_string)
        thread.start_new_thread(os.system,(cmd_string,))
    else:
        print 'Already have BDA output for model {} {} {} {}. Not proceeding unless files moved/renamed.'.format(perception,pragmatics,production,split_type)

def flatten_bda_output(adaptor_types = ['multimodal_pool1','multimodal_conv42','multimodal_fc6', 'human'], verbosity=1)
    h.flatten_param_posterior(adaptor_types = adaptor_types,verbosity=verbosity)

def run_bda_enumerate(simScaling, split_type):
    if not os.path.exists('./enumerateOutput'):
        os.makedirs('./enumerateOutput')
    if not os.path.exists(os.path.join('./enumerateOutput',split_type)):
        os.makedirs(os.path.join('./enumerateOutput',split_type))
    cmd_string = 'webppl BDA-enumerate.wppl --require ./refModule/ -- --simScaling {} --splitType {}'.format(simScaling, split_type)
    print 'Running: {}'.format(cmd_string)
    thread.start_new_thread(os.system,(cmd_string,))

def run_evaluate(perception, pragmatics, production, split_type):
    if not os.path.exists('./evaluateOutput'):
        os.makedirs('./evaluateOutput')  
    out_dir = './evaluateOutput/{}_{}_{}_{}'.format(perception,\
                                                    pragmatics,\
                                                    production,\
                                                    split_type)
    if not os.path.exists(out_dir):
        # os.makedirs(out_dir)
        cmd_string = 'webppl evaluate.wppl --require ./refModule/ -- --paramSetting {}_{}_{} --adaptorType {} --splitType {}'.format(perception, pragmatics, production, perception, split_type)
        print 'Running: {}'.format(cmd_string)
        thread.start_new_thread(os.system,(cmd_string,))
    else:
        print 'Already have evaluation output for model {} {} {} {}. Not proceeding unless files moved/renamed.'.format(perception,pragmatics,production,split_type)

def run_ais(perception, pragmatics, production, split_type, num_samp):
    if not os.path.exists('./aisOutput'):
        os.makedirs('./aisOutput')
    cmd_string = 'webppl BF.wppl --require ./refModule/ -- --perception {} --pragmatics {} --production {} --splitType {}'.format(perception, pragmatics, production, split_type)
    print '{} | Running: {}'.format(num_samp,cmd_string)
    thread.start_new_thread(os.system,(cmd_string,))    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--wppl', type=str, help='options: BDA | BDA-enumerate | evaluate | AIS', default='BDA')
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
    parser.add_argument('--num_ais_samples', type=int, \
                        help='how many AIS samples do you want to take in parallel?',
                        default = 1)

    args = parser.parse_args()
    print args.split_type
    perception = args.perception
    production = args.production
    pragmatics = args.pragmatics
    split_type = args.split_type
    lb = args.sim_scaling_lb
    ub = args.sim_scaling_ub
    if lb==ub: ## to get last value in range, make sure that np.arange has an interval of non-zero length to work with
        ub = ub + 2
    step_size = args.step_size

    assert args.wppl in ['BDA','evaluate', 'BDA-enumerate', 'AIS', 'flatten']

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

    elif 'AIS' in args.wppl:
        for perc in perception:
            for prag in pragmatics:
                for prod in production:
                    for split in split_type:
                        for num_samp in np.arange(args.num_ais_samples):                            
                            run_ais(perc,prag,prod,split,num_samp)   

    elif 'flatten' in args.wppl:
        flatten_bda_output(adaptor_types = ['multimodal_pool1','multimodal_conv42','multimodal_fc6', 'human'], verbosity=1)             

    else:
        print '{} wppl command not recognized'.format(args.wppl)

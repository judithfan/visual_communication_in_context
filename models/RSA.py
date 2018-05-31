from __future__ import division
import os
import thread
import numpy as np

### python RSA.py --wppl BDA --perception multimodal_pool1 multimodal_conv42 multimodal_fc6 human --pragmatics combined S1 S0 --production cost nocost --split_type balancedavg1
### python RSA.py --wppl BDA --perception multimodal_fc6 --pragmatics combined --production cost nocost --split_type balancedavg16 balancedavg32 balancedavg64 balancedavg128 balancedavg256 balancedavg512
### balancedavg3 balancedavg4 balancedavg5

def run_bda(perception, pragmatics, production, split_type):
    cmd_string = 'webppl BDA.wppl --require ./refModule/ -- --perception {} --pragmatics {} --production {} --splitType {}'.format(perception, pragmatics, production, split_type)
    print 'Running: {}'.format(cmd_string)
    thread.start_new_thread(os.system,(cmd_string,))

def run_bda_enumerate(perception, pragmatics, production):
    cmd_string = 'webppl BDA-enumerate.wppl --require ./refModule/ -- --perception {} --pragmatics {} --production {}'.format(perception, pragmatics, production)
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

    args = parser.parse_args()

    perception = args.perception
    production = args.production
    pragmatics = args.pragmatics
    split_type = args.split_type


    assert args.wppl in ['BDA','evaluate', 'BDA-enumerate']

    ## first run BDA.wppl
    if args.wppl=='BDA':
        for perc in perception:
            for prag in pragmatics:
                for prod in production:
                    for split in split_type:
                        run_bda(perc,prag,prod,split)

    ## first run BDA-enumerate.wppl
    if args.wppl=='BDA-enumerate':
        for perc in perception:
            for prag in pragmatics:
                for prod in production:
                    run_bda_enumerate(perc,prag,prod)

    elif args.wppl=='evaluate':
        ## then on output, run evaluate.wppl
        for perc in perception:
            print perc
            for prag in pragmatics:
                for prod in production:
                    run_evaluate(perc,prag,prod,split_type)

    else:
        print '{} wppl command not recognized'.format(args.wppl)

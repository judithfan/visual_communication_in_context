from __future__ import division
import os
import thread
import numpy as np

def run_bda(perception, pragmatics, production):
    cmd_string = 'webppl BDA.wppl --require ./refModule/ -- --perception {} --pragmatics {} --production {}'.format(perception, pragmatics, production)
    print 'Running: {}'.format(cmd_string)
    thread.start_new_thread(os.system,(cmd_string,))

def run_bda_enumerate(perception, pragmatics, production):
    cmd_string = 'webppl BDA-enumerate.wppl --require ./refModule/ -- --perception {} --pragmatics {} --production {}'.format(perception, pragmatics, production)
    print 'Running: {}'.format(cmd_string)
    thread.start_new_thread(os.system,(cmd_string,))

def run_evaluate(perception, pragmatics, production):
    cmd_string = 'webppl evaluate.wppl --require ./refModule/ -- --paramSetting {}_{}_{} --adaptorType {}'.format(perception, pragmatics, production, perception)
    print 'Running: {}'.format(cmd_string)
    thread.start_new_thread(os.system,(cmd_string,))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--wppl', type=str, help='options: BDA | BDA-enumerate | evaluate', default='BDA')
    parser.add_argument('--perception', nargs='+', type=str, \
                        help='option: options: sketch_unroll_full25k | human_full25k | multimodal_full25k',\
                        default = 'human_full25k')
    parser.add_argument('--pragmatics', nargs='+', type=str, \
                        help='option: combined | S1 | S0',\
                        default = 'combined')
    parser.add_argument('--production', nargs='+', type=str, \
                        help='option: cost | nocost',\
                        default = 'cost')
    args = parser.parse_args()

    perception = args.perception
    production = args.production
    pragmatics = args.pragmatics

    assert args.wppl in ['BDA','evaluate', 'BDA-enumerate']

    ## first run BDA.wppl
    if args.wppl=='BDA':
        for perc in perception:
            for prag in pragmatics:
                for prod in production:
                    run_bda(perc,prag,prod)

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
                    run_evaluate(perc,prag,prod)

    else:
        print '{} wppl command not recognized'.format(args.wppl)

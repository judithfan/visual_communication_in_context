from __future__ import division
import os
import thread
import numpy as np

# webppl BDA.wppl --require ./refModule/ -- --perception human_full25k --pragmatics combined --production cost
# webppl BDA.wppl --require ./refModule/ -- --perception human_full25k --pragmatics S1 --production cost
# webppl BDA.wppl --require ./refModule/ -- --perception human_full25k --pragmatics S0 --production cost
# webppl BDA.wppl --require ./refModule/ -- --perception human_full25k --pragmatics combined --production nocost
# webppl BDA.wppl --require ./refModule/ -- --perception human_full25k --pragmatics S1 --production nocost
# webppl BDA.wppl --require ./refModule/ -- --perception human_full25k --pragmatics S0 --production nocost
#
# webppl BDA.wppl --require ./refModule/ -- --perception sketch_unroll_full25k --pragmatics combined --production cost
# webppl BDA.wppl --require ./refModule/ -- --perception sketch_unroll_full25k --pragmatics S1 --production cost
# webppl BDA.wppl --require ./refModule/ -- --perception sketch_unroll_full25k --pragmatics S0 --production cost
# webppl BDA.wppl --require ./refModule/ -- --perception sketch_unroll_full25k --pragmatics combined --production nocost
# webppl BDA.wppl --require ./refModule/ -- --perception sketch_unroll_full25k --pragmatics S1 --production nocost
# webppl BDA.wppl --require ./refModule/ -- --perception sketch_unroll_full25k --pragmatics S0 --production nocost

# webppl evaluate.wppl --require ./refModule/ -- --paramSetting human_full25k_combined_cost
# webppl evaluate.wppl --require ./refModule/ -- --paramSetting human_full25k_S1_cost
# webppl evaluate.wppl --require ./refModule/ -- --paramSetting human_full25k_S0_cost
# webppl evaluate.wppl --require ./refModule/ -- --paramSetting human_full25k_combined_nocost
# webppl evaluate.wppl --require ./refModule/ -- --paramSetting human_full25k_S1_nocost
# webppl evaluate.wppl --require ./refModule/ -- --paramSetting human_full25k_S0_nocost

# webppl evaluate.wppl --require ./refModule/ -- --paramSetting sketch_unroll_full25k_combined_cost
# webppl evaluate.wppl --require ./refModule/ -- --paramSetting sketch_unroll_full25k_S1_cost
# webppl evaluate.wppl --require ./refModule/ -- --paramSetting sketch_unroll_full25k_S0_cost
# webppl evaluate.wppl --require ./refModule/ -- --paramSetting sketch_unroll_full25k_combined_nocost
# webppl evaluate.wppl --require ./refModule/ -- --paramSetting sketch_unroll_full25k_S1_nocost
# webppl evaluate.wppl --require ./refModule/ -- --paramSetting sketch_unroll_full25k_S0_nocost

def run_bda(perception, pragmatics, production):
    cmd_string = 'webppl BDA.wppl --require ./refModule/ -- --perception {} --pragmatics {} --production {}'.format(perception, pragmatics, production)
    print 'Running: {}'.format(cmd_string)
    thread.start_new_thread(os.system,(cmd_string,))

def run_evaluate(perception, pragmatics, production):
    cmd_string = 'webppl evaluate.wppl --require ./refModule/ -- --paramSetting {}_{}_{} --adaptorType {}'.format(perception, pragmatics, production, perception_opts)
    print 'Running: {}'.format(cmd_string)
    thread.start_new_thread(os.system,(cmd_string,))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--wppl', type=str, help='options: BDA | evaluate', default='BDA')
    parser.add_argument('--perception', type=str, help='which generation of sketch photo adaptor? options: sketch_unroll_full25k | human_full25k', default='sketch_unroll_full25k')
    parser.add_argument('--pragmatics', type=str, help='which pragmatics model? options: combined | S1 | S0 ')
    parser.add_argument('--production', type=str, help='options: cost | nocost', default='cost')
    args = parser.parse_args()

    perception_opts = ['sketch_unroll_full25k']
    production_opts = ['cost','nocost']
    pragmatics_opts = ['combined','S1','S0']

    assert args.wppl in ['BDA','evaluate']

    ## first run BDA.wppl
    if args.wppl=='BDA':
        for perc in perception_opts:
            for prag in pragmatics_opts:
                for prod in production_opts:
                    run_bda(perc,prag,prod)

    elif args.wppl=='evaluate':
        ## then on output, run evaluate.wppl
        for perc in perception_opts:
            for prag in pragmatics_opts:
                for prod in production_opts:
                    run_evaluate(perc,prag,prod,perc)

    else:
        print '{} wppl command not recognized'.format(args.wppl)

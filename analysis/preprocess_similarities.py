import os
import shutil
import argparse

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def clean_dirlist(x):
    return [i for i in x if i != '.DS_Store']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_src_similarities', type=str, help='path to dir containing similarities, \
                        multiple crossval splits expected', default='../pix2svg-similarities-053018')
    parser.add_argument('--split_type', type=str, help='type of split and learning objective during adaptor training', default='balancedavg')

    args = parser.parse_args()

    ## destination directories
    path_to_test_examples = './'
    path_to_dest_similarities = '../models/refModule/json/'
    if not os.path.exists(path_to_dest_similarities):
        os.makedirs(path_to_dest_similarities)

    ## get list of layers
    contents = clean_dirlist(os.listdir(args.path_to_src_similarities))
    layers = ['pool1','conv42','fc6']
    split_dir = 'train_test_split'

    ## copy similarities over to the models/refModule/json location, appending the split number to the split_type tag, e.g., "balancedavg1"
    for i,layer in enumerate(layers):
        splits = clean_dirlist(os.listdir(os.path.join(args.path_to_src_similarities,layer)))
        for split_num in splits:
            fname ='similarity-{}{}-multimodal_{}-raw.json'.format(args.split_type,split_num,layer)
            out_path = os.path.join(path_to_dest_similarities,'{}{}'.format(args.split_type,split_num),fname)
            if not os.path.exists(os.path.join(path_to_dest_similarities,'{}{}'.format(args.split_type,split_num))):
                os.makedirs(os.path.join(path_to_dest_similarities,'{}{}'.format(args.split_type,split_num)))
            in_path = os.path.join(args.path_to_src_similarities,layer,split_num,'dump.json')
            print 'Copying {} to {}'.format(in_path, out_path)
            shutil.copy(in_path,out_path)

    ## copy test_examples to this analysis dir
    splits = clean_dirlist(os.listdir(os.path.join(args.path_to_src_similarities,split_dir)))
    for split_num in splits:
        fname = 'pilot2_multimodal_{}{}_test_examples.json'.format(args.split_type,split_num)
        out_path = os.path.join('./',fname)
        in_path = os.path.join(args.path_to_src_similarities,split_dir,split_num,'test_split.json')
        print 'Copying {} to {}'.format(in_path, out_path)
        shutil.copy(in_path,out_path)

        ## save another copy for humans
        fname = 'pilot2_human_{}{}_test_examples.json'.format(args.split_type,split_num)
        out_path = os.path.join('./',fname)
        print 'Copying {} to {}'.format(in_path, out_path)
        shutil.copy(in_path,out_path)

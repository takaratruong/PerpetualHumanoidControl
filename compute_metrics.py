import os
import argparse
import numpy as np
import collections

from uhc.smpllib.smpl_eval import compute_metrics_lite


def compute_metrics(args):
    filenames = os.listdir(args.stats_dir)
    filenames = [f for f in filenames if f.endswith('.npz')]
    stats = collections.defaultdict(list)
    for file in filenames:
        fp = os.path.join(args.stats_dir, file)
        data = np.load(fp, allow_pickle=True)
        for key in data.keys():
            stats[key].extend(data[key])

    success = np.array(stats['success'])
    success_rate = np.mean(success)
    print(f'Success rate: {success_rate}')
    with open(os.path.join(args.stats_dir, 'stats.txt'), 'w') as f:
        for key, val in stats.items():
            if key == 'success':
                continue
            elif key == 'failed_names':
                continue
        
            val = np.array(val, dtype=object)[success]
            val = np.concatenate([v.flatten() for v in val])
            mean = np.mean(val)

            print(f'{key}: {mean}')
            f.write(f'{key}: {mean}\n')
        
        f.write(f'Success rate: {success_rate}\n')
        f.write(f'Num eps: {len(success)}\n\n')

        if 'failed_names' in stats:
            for name in stats['failed_names']:
                f.write(f'{name}\n')
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stats_dir', type=str, required=True)
    args = parser.parse_args()
    compute_metrics(args)
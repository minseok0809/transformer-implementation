
import json
import datetime
from pytz import timezone
import argparse
from prepare_data import load_data

parser = argparse.ArgumentParser(description='Transformers')
parser.add_argument('--config', type=str, default=None)
parser.add_argument('--data_dir', type=str, default=None)

def main(args):

    train_dataset, validation_dataset, test_dataset = load_data(args.data_dir)
    print(train_dataset); print()
    print(validation_dataset); print()
    print(test_dataset); print()


if __name__ == '__main__':

    args = parser.parse_args()
    now_time = str(datetime.datetime.now(timezone('Asia/Seoul')).strftime('%m-%d %H:%M'))
    args.time = now_time

    default_config = vars(args)

    with open(args.config, "w") as f:
        json.dump(default_config, f, indent=0)
    
    main(args)
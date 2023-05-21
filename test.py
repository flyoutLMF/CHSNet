import argparse
import os

from utils.fsc_tester import FSCTester

#  add test fsc
# cl_nonlocal_attention_point_1e-4_0.07
def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--tag', default='cl_nonlocal_attention_point_1e-5_0.07', help='tag of training')
    parser.add_argument('--device', default='0', help='assign device')

    parser.add_argument('--data-dir', default=r'./datasets/FSC', help='training data directory')
    parser.add_argument('--log-param', type=float, default=100.0, help='dmap scale factor')
    parser.add_argument('--dcsize', type=int, default=8, help='divide count size for density map')

    parser.add_argument('--num-workers', type=int, default=4, help='the num of training process')

    parser.add_argument('--save-dir', default='./test_result', help='directory to save models.')
    parser.add_argument('--resume', default='./checkpoint/cl_nonlocal_attention_point_1e-5_0.07/best_model.pth',
                        help='the path of resume training model')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu

    tester = FSCTester(args)
    tester.setup()
    tester.test()


import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Go CIPS")
    parser.add_argument('--bpr_batch', type=int, default=2048,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--recdim', type=int, default=256,
                        help="the embedding size of cips")
    parser.add_argument('--layer', type=int, default=3,
                        help="the layer num of CIPS")
    parser.add_argument('--lr', type=float, default=0.001,
                        help="the learning rate")
    parser.add_argument('--decay', type=float, default=1e-4,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--dropout', type=int, default=1,
                        help="using the dropout or not")
    parser.add_argument('--keepprob', type=float, default=0.6,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--a_fold', type=int, default=100,
                        help="the fold num used to split large adj matrix, like gowalla")
    parser.add_argument('--testbatch', type=int, default=2048,
                        help="the batch size of users for testing")
    parser.add_argument('--dataset', type=str, default='gowalla',
                        help="available datasets: [lastfm, gowalla, yelp2018, amazon-book]")
    parser.add_argument('--path', type=str, default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--topks', nargs='?', default="[20]",
                        help="@k test list")
    parser.add_argument('--tensorboard', type=int, default=1,
                        help="enable tensorboard")
    parser.add_argument('--comment', type=str, default="cips")
    parser.add_argument('--load', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
    parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('--model', type=str, default='cips', help='rec-model, support [mf, cips]')
    parser.add_argument('--L', type=float, default=0, help='L-GCN(r=?)')
    parser.add_argument('--GPU', type=int, default=0, help='GPU')
    parser.add_argument('--lam1', type=float, default=0., help='user_w')
    parser.add_argument('--lam2', type=float, default=0., help='item_w')
    parser.add_argument('--fix', type=float, default=0., help='fix')
    parser.add_argument('--fix0', type=float, default=0.55, help='fix0')
    return parser.parse_args()

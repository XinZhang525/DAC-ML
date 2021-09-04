import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--target", type=str, default="speed", choices=["speed", "inflow", "demand"])

    parser.add_argument("--region_i", type=int, default=15, help="i of region index.")
    parser.add_argument("--region_j", type=int, default=20, help="j of region index.")
    parser.add_argument("--num_layers", type=int, default=1, help="num layers of lstm in encoder.")
    parser.add_argument("--D_out", type=int, default=128, help="out features.")
    parser.add_argument("--z_dim", type=int, default=128, help="latent dimension.")

    parser.add_argument("--task_length", type=int, default=24, help="latent dimension.")
    parser.add_argument("--WINDOW_SIZE", type=int, default=5, help="rolling window size of each sub task.")

    parser.add_argument("--task_iteration", type=int, default=1000, help="number of tasks sampled for training.")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate.")
    parser.add_argument("--beta1", type=float, default=0.5, help="adam: decay of first order momentum of gradient.")
    parser.add_argument("--beta2", type=float, default=0.999, help="adam: decay of first order momentum of gradient.")
    parser.add_argument("--seed", type=int, default=0, help="random seed.")
    
    parser.add_argument('--domains', type=str, nargs='+',
                    default=['daily'],
                    help='characterizes the domain information to capture')
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--ckpt', type=str, default="")
    parser.add_argument('--time', type=str, default="")
    parser.add_argument('--region_is', type=str, nargs='+')
    

    parser.add_argument("--log_wandb", type=int, default=1)
    
    opt = parser.parse_args()
    
    return opt
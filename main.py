import os
import sys
import pdb
import yaml
from utils.train_utils import *
from utils.dataset_utils import update_num_attributes
from cluster import cluster
import wandb


def parse_config():
    def boolean_string(s):
        if s not in {'False', 'True'}:
            raise ValueError('Not a valid boolean string')
        return s == 'True'

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='cub_bn.yaml', help='configurations for training')
    parser.add_argument("--num_attributes", help='number of attributes')
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--eps", default=2, type=float)
    parser.add_argument("--t", default=5, type=int)
    parser.add_argument("--lam", default=1, type=float)

    # regularization
    parser.add_argument("--score_reg", default=True, type=boolean_string)
    parser.add_argument("--euclidean_reg", default=False, type=boolean_string,
                        help='use euclidean distance instead of mahalanobis distance')
    parser.add_argument("--mahalanobis_reg", default=False, type=boolean_string,
                        help='constrain the use of mahalanobis distance even in the datasets without')
    parser.add_argument("--reg_type", default=1, type=int,
                        help="1 = score, 2 = mahalanobis, 3 = euclidean, 4 = None. If type is 1/2/3/4, it overrides the default setting")
    parser.add_argument("--selection_method", default=1, type=int, help="1 = hungarian, 2 = nearest neighbor, 3 = random")
    parser.add_argument("--speed_hungarian", default=True, type=boolean_string)
    parser.add_argument("--linear_probe", default=False, type=boolean_string)
    # num_attributes
    parser.add_argument("--outdir", default='./outputs', help='where to put all the results')
    parser.add_argument("--save_chosen_concepts", default=False, type=boolean_string)
    parser.add_argument("--save_attribute_embeddings", default=False, type=boolean_string)

    return parser.parse_args()


def main(cfg, args):
    set_seed(args.seed)
    print(cfg)
    update_num_attributes(args)
    best_model_path = "models/" + str(cfg['dataset']) + "_" + str(args.num_attributes) + "-concepts"
    if args.reg_type == 1 and args.selection_method == 1:
        best_model_path += "_our-method"
    best_model_path += "_best-model.pth"

    if args.selection_method in {1, 2} and args.num_attributes != 'full':
        if args.linear_probe:
            acc, model = cluster(cfg, args)
        else:
            acc, model, attributes, attributes_embeddings = cluster(cfg, args)
    else:
        attributes, attributes_embeddings = cluster(cfg, args)

    if args.linear_probe:
        return acc, model

    if args.save_chosen_concepts:
        with open(f"selectedconcepts/{cfg['dataset']}_{args.num_attributes}_chosen_concepts.txt", "w") as f:
            for attr in attributes:
                f.write(f"{attr}\n")
    # get attributes back from the file with each line as an attribute


    if cfg['reinit'] and args.num_attributes != 'full' and args.selection_method in {1, 2}:
        # assert cfg['cluster_feature_method'] == 'linear'
        feature_train_loader, feature_test_loader = get_feature_dataloader(cfg)
        model[0].weight.data = attributes_embeddings.cuda() * model[0].weight.data.norm(dim=-1, keepdim=True)
        # create random weight for the last layer
        # model[1].weight.data = torch.randn(model[1].weight.data.shape).cuda()
        for param in model[0].parameters():
            param.requires_grad = False
        best_model, best_acc = train_model(args, cfg, cfg['epochs'], model, feature_train_loader, feature_test_loader,
                                           first=False)

    else:
        model = get_model(args, cfg, cfg['score_model'], input_dim=len(attributes),
                          output_dim=get_output_dim(cfg['dataset']))
        score_train_loader, score_test_loader = get_score_dataloader(cfg, attributes_embeddings)
        best_model, best_acc = train_model(args, cfg, cfg['epochs'], model, score_train_loader, score_test_loader,
                                           first=False)

    # save best model
    torch.save(best_model.state_dict(), best_model_path)

    return best_model, best_acc


if __name__ == '__main__':

    args = parse_config()

    with open(f"{args.config}", "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    main(cfg, args)

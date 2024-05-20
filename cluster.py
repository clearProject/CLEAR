from utils.train_utils import *
# import open_clip
from sklearn.cluster import KMeans
import numpy as np
import torch
from utils.train_utils import *
from hungarian import hungarian_algorithm
import wandb
from score_based_concepts import train_score_model, sample_points
import time


def cluster(cfg, args):
    run_name = str(cfg['dataset']) + '_' + str(args.num_attributes) + "concepts"
    run = wandb.init(project="CLEAR", config=cfg, name=run_name)

    if cfg['model_type'] == 'clip':
        model, preprocess = clip.load(cfg['model_size'])
    else:
        raise NotImplementedError

    attributes = get_attributes(cfg, args)
    attribute_embeddings = []
    batch_size = 32
    for i in range((len(attributes) // batch_size) + 1):
        sub_attributes = attributes[i * batch_size: (i + 1) * batch_size]
        clip_attributes_embeddings = None
        if cfg['model_type'] == 'clip':
            clip_attributes_embeddings = clip.tokenize([get_prefix(cfg) + attr for attr in sub_attributes]).cuda()

        attribute_embeddings += [embedding.detach().cpu() for embedding in
                                 model.encode_text(clip_attributes_embeddings)]
        
    attribute_embeddings = torch.stack(attribute_embeddings).float()
    attribute_embeddings = attribute_embeddings / attribute_embeddings.norm(dim=-1, keepdim=True)

    if args.save_attribute_embeddings:
        # save the attribute embeddings tensor
        save_path = str(cfg['dataset']) + "_attribute_embeddings.pt"
        torch.save(attribute_embeddings, save_path)

    print("bank size: " + str(len(attribute_embeddings.numpy())))

    print("num_attributes: ", args.num_attributes)
    if args.num_attributes == 'full':
        return attributes, attribute_embeddings

    if args.selection_method == 3:  # random
        selected_idxes = np.random.choice(np.arange(len(attribute_embeddings)), size=args.num_attributes, replace=False)
    else:
        if args.selection_method == 1 or args.selection_method == 2:  # hungarian of nearest neighbor
            mu = torch.mean(attribute_embeddings, dim=0)
            sigma_inv = torch.tensor(np.linalg.inv(torch.cov(attribute_embeddings.T)))
            var = torch.var(attribute_embeddings, dim=0)  # edit
            configs = {
                'mu': mu,
                'sigma_inv': sigma_inv,
                'mean_distance': np.mean([mahalanobis_distance(embed, mu, sigma_inv) for embed in attribute_embeddings]),
                'var': var,  # edit
                'attribute_embeddings': attribute_embeddings
            }

            model = get_model(args, cfg, cfg['linear_model'], input_dim=attribute_embeddings.shape[-1], output_dim=get_output_dim(cfg['dataset']))
            if args.linear_probe:
                model = get_model(args, cfg, cfg['score_model'], input_dim=attribute_embeddings.shape[-1],
                                  output_dim=get_output_dim(cfg['dataset']))
            model.cuda()
            train_loader, test_loader = get_feature_dataloader(cfg)
            # score_based = True
            atts_data_loader = torch.utils.data.DataLoader(attribute_embeddings, batch_size=batch_size, shuffle=True)
            if not args.linear_probe:
                score_model = train_score_model(attribute_embeddings, atts_data_loader, train_loader)
            else:
                score_model = None

            regularizer = None
            if args.reg_type == 1:
                regularizer = 'score'
            elif args.reg_type == 2:
                regularizer = 'mahalanobis'
            elif args.reg_type == 3:
                regularizer = 'euclidean'
            elif args.reg_type == 4:
                regularizer = None
            else:
                if args.score_reg:
                    regularizer = 'score'
                elif args.mahalanobis_reg:
                    regularizer = 'mahalanobis'
                elif args.euclidean_reg:
                    regularizer = 'euclidean'
            if args.linear_probe:
                regularizer = None

            if regularizer != None:
                best_model, best_acc = train_model(args, cfg, cfg['linear_epochs'], model, train_loader,
                                                   test_loader, score_model, regularizer=regularizer, configs=configs,
                                                   first=True)
            else:
                if cfg.get("cosine", False):
                    best_model, best_acc = train_model(args, cfg, cfg['linear_epochs'], model, train_loader,
                                                       test_loader, score_model, regularizer='cosine', configs=configs,
                                                       first=True)

                else:
                    best_model, best_acc = train_model(args, cfg, cfg['linear_epochs'], model, train_loader,
                                                       test_loader, score_model, regularizer=None, configs=configs, first=True)

            if args.linear_probe:
                return best_acc, best_model

            centers = best_model[0].weight.detach().cpu().numpy()

        selected_idxes = []
        if args.selection_method == 2:
            for center in centers:
                center = center / torch.tensor(center).norm().numpy()
                distances = np.sum((attribute_embeddings.numpy() - center.reshape(1, -1)) ** 2, axis=1)
                # sorted_idxes = np.argsort(distances)[::-1]
                sorted_idxes = np.argsort(distances)
                count = 0
                while sorted_idxes[count] in selected_idxes:
                    count += 1
                selected_idxes.append(sorted_idxes[count])
        elif args.selection_method == 1:
            print("Using Hungarian Algorithm")
            start_time = time.time()
            if args.speed_hungarian:
                # selected_fast_idxes_set = set()
                print("Sped up")
                sped_up_successfuly = True
                reduction_num = 5

                def reduce_pool(centers, reduction_num):
                    selected_fast_idxes_set = set()
                    for center in centers:
                        center = center / torch.tensor(center).norm().numpy()
                        distances = np.sum((attribute_embeddings.numpy() - center.reshape(1, -1)) ** 2, axis=1)
                        sorted_idxes = np.argsort(distances)
                        # take the top 5
                        # selected_idxes.append(sorted_idxes[:reduction_num])
                        selected_fast_idxes_set.update(set(sorted_idxes[:reduction_num]))
                    selected_fast_idxes = list(selected_fast_idxes_set)
                    print("For m = " + str(reduction_num) + ", a smaller pool obtained with size: "
                          + str(len(selected_fast_idxes)))
                    return selected_fast_idxes

                selected_fast_idxes = reduce_pool(centers, reduction_num)
                while len(selected_fast_idxes) < len(centers):
                    reduction_num = reduction_num * 2
                    selected_fast_idxes = reduce_pool(centers, reduction_num)

                selected_concepts_for_reduction = [attributes[i] for i in selected_fast_idxes]
                selected_embeddings_for_reduction = torch.tensor(attribute_embeddings[selected_fast_idxes])
                # update the current attributes
                attributes = selected_concepts_for_reduction
                attribute_embeddings = selected_embeddings_for_reduction
                wandb.log({"sped_successfuly": sped_up_successfuly})
                wandb.log({"m": reduction_num})

            distances_array = np.zeros((len(attribute_embeddings.numpy()), len(attribute_embeddings.numpy())))
            # create distance array
            for i, center in enumerate(centers):
                # cosine similarity
                center = center / torch.tensor(center).norm().numpy()
                # calculate the dot product between the centers and the attribute embeddings
                distances = [np.dot(center, embed) for embed in attribute_embeddings.numpy()]
                distances_array[i, :] = distances

            # if we want max instead of min
            max_value = np.max(distances_array)
            distances_diff = max_value - distances_array
            ans_pos = hungarian_algorithm(distances_diff.copy())
            # if we want min
            # ans_pos = hungarian_algorithm(distances_array.copy())

            # the selected indexes are the first len(centers) columns in ans_pos
            for i in range(len(centers)):
                selected_idxes.append(ans_pos[i][1])
            print("Finished selecting concepts")
            end_time = time.time()
            time_in_minutes = (end_time - start_time) / 60
            wandb.log({"hungarian_runtime": time_in_minutes})
        selected_idxes = np.array(selected_idxes)
        # find the sum of distances between the selected concepts

    if args.selection_method in {1, 2}:
        return best_acc, best_model, [attributes[i] for i in selected_idxes], torch.tensor(attribute_embeddings[selected_idxes])
    else:
        return [attributes[i] for i in selected_idxes], torch.tensor(attribute_embeddings[selected_idxes])

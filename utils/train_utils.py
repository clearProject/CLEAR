'''
evaluate zero-shot performance
'''
import copy
import time
import wandb
# import open_clip
from dataset import FeatureDataset, OnlineScoreDataset
from utils.dataset_utils import *
from scipy.spatial import distance
import numpy as np
import torch
from score_based_concepts import sample_from_known_points


def set_seed(seed):
    if seed == -1:
        seed = int(torch.randint(0, 2 ** 32 - 1, (1,)).item())

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_model(args, cfg, model, input_dim, output_dim):
    if args.num_attributes == 'full':
        num_attributes = len(get_attributes(cfg, args))
    else:
        num_attributes = args.num_attributes

    if model == ['linear', 'bn', 'linear']:
        model = nn.Sequential(
            nn.Linear(input_dim, num_attributes, bias=False),
            nn.BatchNorm1d(num_attributes),
            nn.Linear(num_attributes, output_dim)
        )
    elif model == ['bn', 'linear']:
        model = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, output_dim, bias=False),
        )
    elif model == ['linear', 'linear']:
        model = nn.Sequential(
            nn.Linear(input_dim, num_attributes, bias=False),
            nn.Linear(num_attributes, output_dim)
        )
    elif model == ['linear']:
        model = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=False),
        )

    else:
        raise NotImplementedError

    return model


def get_feature_dataloader(cfg):
    if cfg['model_type'] == 'clip':
        model, preprocess = clip.load(cfg['model_size'])
    else:
        raise NotImplementedError

    train_loader, test_loader = get_image_dataloader(cfg['dataset'], preprocess)

    train_features = get_image_embeddings(cfg, cfg['dataset'], model, train_loader, 'train')
    test_features = get_image_embeddings(cfg, cfg['dataset'], model, test_loader, 'test')

    if cfg['dataset'] == 'imagenet-animal':
        # train_labels, test_labels = get_labels("imagenet")
        train_labels, test_labels = get_labels("imagenet-animal")
        train_labels, test_labels = np.array(train_labels), np.array(test_labels)

        train_idxes = np.where((train_labels < 398) & (train_labels != 69))
        train_features = train_features[train_idxes]

        test_idxes = np.where((test_labels < 398) & (test_labels != 69))
        test_features = test_features[test_idxes]

    train_labels, test_labels = get_labels(cfg['dataset'])
    train_score_dataset = FeatureDataset(train_features, train_labels)
    test_score_dataset = FeatureDataset(test_features, test_labels)

    train_loader = DataLoader(train_score_dataset, batch_size=cfg['batch_size'], shuffle=True)
    test_loader = DataLoader(test_score_dataset, batch_size=cfg['batch_size'], shuffle=False)
    return train_loader, test_loader


def get_score_dataloader(cfg, attribute_embeddings):
    if cfg['model_type'] == 'clip':
        model, preprocess = clip.load(cfg['model_size'])
    else:
        raise NotImplementedError
    train_loader, test_loader = get_image_dataloader(cfg['dataset'], preprocess)
    print("Get Embeddings...")
    train_features = get_image_embeddings(cfg, cfg['dataset'], model, train_loader, 'train')
    test_features = get_image_embeddings(cfg, cfg['dataset'], model, test_loader, 'test')
    if cfg['dataset'] == 'imagenet-animal':
        train_labels, test_labels = get_labels("imagenet")
        train_labels, test_labels = np.array(train_labels), np.array(test_labels)

        train_idxes = np.where((train_labels < 398) & (train_labels != 69))
        train_features = train_features[train_idxes]

        test_idxes = np.where((test_labels < 398) & (test_labels != 69))
        test_features = test_features[test_idxes]

    train_labels, test_labels = get_labels(cfg['dataset'])

    print("Initializing Feature Dataset")
    train_feature_dataset = FeatureDataset(train_features, train_labels)
    test_feature_dataset = FeatureDataset(test_features, test_labels)
    train_loader = DataLoader(train_feature_dataset, batch_size=cfg['batch_size'], shuffle=False)
    test_loader = DataLoader(test_feature_dataset, batch_size=cfg['batch_size'], shuffle=False)

    train_scores = extract_concept_scores(train_loader, model, attribute_embeddings)
    test_scores = extract_concept_scores(test_loader, model, attribute_embeddings)

    train_score_dataset = FeatureDataset(train_scores, train_labels)
    test_score_dataset = FeatureDataset(test_scores, test_labels)

    train_loader = DataLoader(train_score_dataset, batch_size=cfg['batch_size'], shuffle=True)
    test_loader = DataLoader(test_score_dataset, batch_size=cfg['batch_size'], shuffle=False)

    return train_loader, test_loader


def train_model(args, cfg, epochs, model, train_loader, test_loader, score_net=None, regularizer=None, configs=None, first=True):
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    loss_function = torch.nn.CrossEntropyLoss()

    best_acc = 0
    best_worst_group_acc = 0
    last_best_acc = None
    best_model = copy.deepcopy(model)
    no_break = True

    """
    # if it is not their run, double the epochs
    if not args.use_hungarian and not args.labo_selected_concepts:
        epochs = epochs
    else:
        if not first:
            epochs = epochs * 2
    if args.constrain_mah and first:
        regularizer = 'mahalanobis'
    """

    for epoch in range(epochs):
        # train:
        total_hit = 0
        total_num = 0
        for idx, batch in enumerate(train_loader):
            s, t = batch[0], batch[1]
            s = s.float().cuda()
            t = t.long().cuda()
            output = model(s)
            loss = loss_function(output, t)

            if regularizer == 'score':
                weights = model[0].weight / model[0].weight.data.norm(dim=-1, keepdim=True)
                score_samples = sample_from_known_points(weights, score_net, args.eps, args.t)
                # normalize the weights
                score_samples = score_samples / torch.norm(score_samples, dim=-1, keepdim=True)
                # The norm of each row diff
                scores_norm = torch.sum((score_samples - weights) ** 2, dim=1)
                scores_loss = torch.mean(scores_norm)
                loss += args.lam * torch.abs(scores_loss)

            elif regularizer == 'euclidean':
                euclidean_loss = 0
                for att in model[0].weight / model[0].weight.data.norm(dim=-1, keepdim=True):
                    te = att - configs['attribute_embeddings'].cuda()
                    sum = torch.sum(te ** 2, 1)
                    mean = torch.mean(sum)
                    euclidean_loss += mean
                loss += euclidean_loss

            elif regularizer == 'mahalanobis':
                mahalanobis_loss = (mahalanobis_distance(
                    model[0].weight / model[0].weight.data.norm(dim=-1, keepdim=True), configs['mu'].cuda(),
                    configs['sigma_inv'].cuda()) - configs['mean_distance']) / (
                                           configs['mean_distance'] ** args.division_power)
                loss += torch.abs(mahalanobis_loss)
                
            total_hit += torch.sum(t == output.argmax(-1))
            total_num += len(t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if first:
                wandb.log({"first_loss": loss})
            else:
                wandb.log({"second_loss": loss})

        # test:
        with torch.no_grad():
            group_array = []
            predictions = []
            labels = []
            for idx, batch in enumerate(test_loader):
                s, t = batch[0], batch[1]
                s = s.float().cuda()
                output = model(s).cpu()
                pred = torch.argmax(output, dim=-1)
                if len(batch) == 3:
                    group_array.append(batch[2])
                predictions.append(pred)
                labels.append(t)

            predictions = torch.cat(predictions)
            if len(group_array) > 0:
                group_array = torch.cat(group_array)

        labels = torch.cat(labels)

        acc = (torch.sum(predictions == labels) / len(predictions) * 100)

        if acc > best_acc:
            best_acc = acc
            best_model = copy.deepcopy(model)

        if first:
            wandb.log({"first_accuracy": acc})
            wandb.log({"max_first_acc": best_acc})
        else:
            wandb.log({"second_accuracy": acc})
            wandb.log({"max_accuracy": best_acc})

        if epoch % 10 == 0:
            print(f"Epoch [{epoch}], Best accuracy:", best_acc.item(), "Last accuracy:", acc.item())

            sys.stdout.flush()

            if not no_break and (last_best_acc is not None and best_acc == last_best_acc):
                break

            last_best_acc = best_acc

    return best_model, best_acc


def mahalanobis_distance(x, mu, sigma_inv):
    x = x - mu.unsqueeze(0)
    return torch.diag(x @ sigma_inv @ x.T).mean()


def get_image_embeddings(cfg, dataset, model, loader, mode='train'):
    if dataset == 'imagenet-a' and mode == 'train':
        folder_name = get_folder_name("imagenet")
    else:
        folder_name = get_folder_name(dataset)

    model_name = cfg['model_type'] + '_' + cfg['model_size'].split("/")[-1]

    if model_name == 'clip_32':
        filename = f"./data/{folder_name}/{mode}_embeddings.npy"
    else:
        filename = f"./data/{folder_name}/{model_name}_{mode}_embeddings.npy"

    if os.path.exists(filename):
        features = np.load(filename)
    else:
        print("Extract and pre-save image features...")
        with torch.no_grad():
            features = []
            for i, batch in tqdm(enumerate(loader), total=len(loader)):
                (images, target) = batch[0], batch[1]
                # images: [batch_size, 3, 224, 224]
                images = images.cuda()
                target = target.cuda()
                image_features = model.encode_image(images)
                # [batch_size, 768]
                image_features /= image_features.norm(dim=-1, keepdim=True)
                # [batch_size, 768]
                features.append(image_features.cpu())
            features = torch.cat(features)
        features = np.array(features)
        np.save(filename, features)

    return features


def extract_concept_scores(loader, model, attribute_embeddings):
    with torch.no_grad():
        scores = []

        for i, (image_features, target) in tqdm(enumerate(loader), total=len(loader)):
            image_features = image_features.cuda().float()
            # target = target.cuda()
            # image_features = model.encode_image(images).float()
            # image_features /= image_features.norm(dim=-1, keepdim=True)

            logits = image_features @ attribute_embeddings.float().T.cuda()
            scores.extend(logits.cpu().to(torch.float16).tolist())

    return scores

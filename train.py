from sys import argv
from unittest import result
import numpy as np
import pickle
import glob
import random
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchmetrics import AUROC, PrecisionRecallCurve, ROC, ConfusionMatrix
import matplotlib.pyplot as plt
import datetime
import os
import pandas as pd
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import argparse
from checkpoint_utils import save_checkpoint, maybe_resume, keep_last_n_checkpoints




#Get this from Zenodo
label_dict = pickle.load(open("multiclass_labels.pkl", "rb"))


#ensembles = 10
k = 10
thresholds = [0, 1, 3, 10]

n_classes = len(thresholds)
threshold_class = n_classes - 2

#class_frequencies = np.zeros(60)

#for labelset in label_dict.values():
    #for cl in range(60):
        #class_frequencies[cl] += np.sum(labelset == cl)

#class_frequencies /= np.sum(class_frequencies)
#print(class_frequencies)
#class_weights = 1 / class_frequencies




class DistogramDataset(Dataset):
    def __init__(self, labels, pkl_dir, n_classes, one_hot=True, all_pairs=False, two_d=False, n_samples_per_target=1, half=False,use_pae=True,sample_pkls=True,sample_pattern='*',parallel_load=True): 
        self.labels = labels
        self.one_hot = one_hot
        self.n_classes = n_classes
        self.two_d = two_d
        self.all_pairs = all_pairs if not self.two_d else True
        self.normalize = True
        self.n_samples_per_target = n_samples_per_target
        self.use_pae = use_pae
        self.masks = dict()
        self.targets = list(labels.keys())
        self.pkl_dir = pkl_dir
        #self.pkl_paths = [random.sample(glob.glob(f"{self.pkl_dir}/{self.targets[idx]}/result*.pkl*"), self.n_samples_per_target) for idx in range(len(self.targets))]
        print('Finding pkls...')
        result = subprocess.run(["find", self.pkl_dir, "-path", f"*/*/afsample2/result{sample_pattern}.pkl"], capture_output=True,text=True)
        pkls = result.stdout.strip().split("\n") if result.stdout.strip() else []
        print(f"Found {len(pkls)} pkls in {self.pkl_dir}")
        #print(pkls)
        target_pkls = []
        for target in self.targets:
            target_pkls.append([pkl for pkl in pkls if target in pkl])

        if sample_pkls:
            rng = random.Random(42)
            #self.pkl_paths = [random.sample(glob.glob(f"{self.pkl_dir}/{self.targets[idx]}/afsample2/result*.pkl"), self.n_samples_per_target) for idx in range(len(self.targets))]
            self.pkl_paths = [rng.sample(target_pkls[idx], self.n_samples_per_target) for idx in range(len(self.targets))]
        else:
            # self.pkl_paths = [glob.glob(f"{self.pkl_dir}/{self.targets[idx]}/afsample2/result*.pkl")[0:1] for idx in range(len(self.targets))]
            self.pkl_paths = [target_pkls[idx] for idx in range(len(self.targets))]



        self.all_dgram_logits = []
        print(len(self.pkl_paths), len(self.targets))


        # Parallel pickle loading
        if parallel_load:
            self.all_dgram_logits = [None] * len(self.pkl_paths)
            with ThreadPoolExecutor(max_workers=min(64, len(self.pkl_paths))) as executor:
                futures = {
                    executor.submit(self._load_target, idx, paths, half): idx
                    for idx, paths in enumerate(self.pkl_paths)
                }
                for future in as_completed(futures):
                    idx, result = future.result()
                    self.all_dgram_logits[idx] = result
        else:   
            for idx, pkl_paths_this_target in enumerate(self.pkl_paths):
                inputs_this_target = []
                print(f"{idx} Processing target {self.targets[idx]} with {len(pkl_paths_this_target)} pkls")
                for pkl_path in pkl_paths_this_target:
                    pkl = pickle.load(open(pkl_path, "rb"))
                    dgram = pkl["distogram"]["logits"]
                    if self.normalize:
                        dgram /= 10.0
                    if half:
                        dgram = dgram.astype("float16")

                    if self.use_pae and "predicted_aligned_error" in pkl:
                        pae = pkl["predicted_aligned_error"][..., None]
                        if half:
                            pae = pae.astype("float16")

                        dgram = np.concatenate((dgram, pae), axis=-1)
                    inputs_this_target.append(dgram)

                self.all_dgram_logits.append(inputs_this_target)
        for key, labels in self.labels.items():
            self.masks[key] = 1 - (labels == -1)

            quantized_labels = np.zeros_like(labels)
            #labels[labels >= (self.n_classes-1)] = self.n_classes - 1
            for i, thr in enumerate(thresholds):
                quantized_labels[labels >= thr] = i
            quantized_labels[labels == -1] = -1

            if self.one_hot:
                if self.all_pairs:
                    quantized_labels[quantized_labels == -1] = 0
                quantized_labels = np.eye(self.n_classes)[quantized_labels.astype(int)]
            self.labels[key] = quantized_labels

        # Build flat index: (target_idx, sample_idx)
        self.flat_index = [
            (i, j)
            for i, dgrams in enumerate(self.all_dgram_logits)
            for j in range(len(dgrams))
        ]
    #def __len__(self): 
    #    n_labels = len(self.all_dgram_logits)
    #    return n_labels
    def __len__(self): 
        return len(self.flat_index)

    def __getitem__(self, idx):
        target_idx, sample_idx = self.flat_index[idx]
        target = self.targets[target_idx]
        dgram_logits = self.all_dgram_logits[target_idx][sample_idx]
        pkl_path = self.pkl_paths[target_idx][sample_idx]
        #target = self.targets[idx]
        #print(len(self.all_dgram_logits), self.n_samples_per_target,idx)
        #dgram_logits = random.choice(self.all_dgram_logits[idx])

        labels = self.labels[target]
        assert dgram_logits.shape[0] == labels.shape[0], "Mismatch in distogram-labels shape"

        if not self.two_d:
            # 2D -> 1D
            dgram_logits = dgram_logits.reshape((-1, 64))
            labels = labels.reshape((-1))

        if not self.all_pairs:
            labels0 = np.where(labels == 0)[0]
            labels1 = np.where(labels == 1)[0]
            labels2 = np.where(labels == 2)[0]
            if labels0.shape[0] > 0:
                print(f"Sampling {labels0.shape[0]} from {labels1.shape[0]} and {labels2.shape[0]} for target {target}")
                labels1 = np.random.choice(labels1, labels0.shape[0])
                labels2 = np.random.choice(labels2, labels0.shape[0])
                labels = labels[np.concatenate((labels0, labels1, labels2))]
                dgram_logits = dgram_logits[np.concatenate((labels0, labels1, labels2))]
            else:
                dgram_logits = dgram_logits[labels>=0]
                labels = labels[labels>=0]
            mask = None

        else:
            mask = self.masks[target]

        return dgram_logits, labels, mask, pkl_path
    
    def _load_pkl(self, pkl_path, half):
        pkl = pickle.load(open(pkl_path, "rb"))
        dgram = pkl["distogram"]["logits"]
        if self.normalize:
            dgram /= 10.0
        if half:
            dgram = dgram.astype("float16")
        if self.use_pae and "predicted_aligned_error" in pkl:
            pae = pkl["predicted_aligned_error"][..., None]
            if half:
                pae = pae.astype("float16")
            dgram = np.concatenate((dgram, pae), axis=-1)
        return dgram

    def _load_target(self, idx, pkl_paths_this_target, half):
        print(f"{idx} Processing target {self.targets[idx]} with {len(pkl_paths_this_target)} pkls")
        inputs_this_target = [self._load_pkl(pkl_path, half) for pkl_path in pkl_paths_this_target]
        return idx, inputs_this_target
    
 


class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        x = x.permute(self.dims)
        return x


class MLP(nn.Module):
    def __init__(self, n_classes, in_channels=65):
        super().__init__()
        self.n_classes = n_classes
        self.layers = nn.Sequential(
            nn.Linear(in_channels, 32),
            nn.Tanh(),
            #nn.Dropout(p=0.2),
            nn.Linear(32, 8),
            nn.Tanh(),
            #nn.Dropout(p=0.2),
            nn.Linear(8, self.n_classes) # contact, no-contact, switch
        )

    def forward(self, x):
        return self.layers(x)


class Conv(nn.Module):
    def __init__(self, n_classes,in_channels=65):
        super().__init__()
        self.n_classes = n_classes
        self.layers = nn.Sequential(
            nn.Linear(in_channels, 32),
            nn.Tanh(),
            nn.Linear(32, 8),
            nn.Tanh(),
            Permute(dims=[0, 3, 1, 2]), # batch, L1, L2, 64 -> batch, 64, L1, L2
            nn.Conv2d(8, 16, (5, 5), padding="same"),
            nn.Tanh(),
            Permute(dims=[0, 2, 3, 1]), # batch, 2, L1, L2 -> batch, L1, L2, 8
            nn.Linear(16, self.n_classes),
        )

    def forward(self, x):
        return self.layers(x)


def plot_pr(precision, recall):
    plt.plot(recall.cpu()[:-1], precision.cpu()[:-1])
    plt.title(f"Target {target}, epoch {e}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig(f"plots/{target}_{e}.png")
    plt.clf()

parser = argparse.ArgumentParser()
parser.add_argument('target')
parser.add_argument('--pkl-dir', default='/proj/wallner-b/users/x_bjowa/distogram_training/cfold2_trimmed/AF_models_dropout/')
parser.add_argument('--training-to-use', type=int, default=10)
parser.add_argument('--half-precision', action='store_true', default=True)
parser.add_argument('--use-pae', action='store_true')
parser.add_argument('--network_type', type=str, default='2dconv')
parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()

target         = args.target
pkl_dir        = args.pkl_dir
training_to_use = args.training_to_use
half_precision = args.half_precision
device         = args.device
use_pae        = args.use_pae
network_type = args.network_type

in_channels = 65 if use_pae else 64
nopae = '_no_pae' if not use_pae else ''


model_cls = Conv if network_type == '2dconv' else MLP
model = model_cls(n_classes=n_classes, in_channels=in_channels)
if half_precision:
    model = model.half()
model = model.cuda()




x10_pattern='*pred_[1,2]_*'  # select 2 pdbs for each of 5 AF2 networks, total 10, 
                                                        # n_samples_per_target should be the same the total number
                                                        # since sample_pkls is True, if False the n_samples_per_target 
                                                        # is not used and all matching the sample_pattern are selected.
x5_pattern='*pred_[1]_*'  # select 1 pdb for each of 5 AF2 networks, total 5
#Will use all samples for training not just a random one per target. n_samples_per_target more data.
#sample_pattern,n_samples_per_target,prefix='*',20,'2dconv_multi_x1_ensemble_all'  #used to be default training. This will however not be the same when restarting från checkpoint
# sample_pattern,n_samples_per_target,prefix=x5_pattern,5,'2dconv_multi_x5_ensemble_all' 

if training_to_use == 10:
    sample_pattern,n_samples_per_target,prefix=x10_pattern,10,f'{network_type}_multi_x10_ensemble_all_fix_aucpr{nopae}' 
    epochs = 300
elif training_to_use == 5:
    sample_pattern,n_samples_per_target,prefix=x5_pattern,5,f'{network_type}_multi_x5_ensemble_all{nopae}'
    epochs = 300
elif training_to_use == 1:
    sample_pattern,n_samples_per_target,prefix='*',1,f'{network_type}_multi_x1_ensemble_all{nopae}'
    epochs = 300
else:
    print(f"Invalid training_to_use value {training_to_use}, should be 1, 5, or 10")
    sys.exit()




print(f"Using sample pattern {sample_pattern} with n_samples_per_target {n_samples_per_target} and prefix {prefix}")
#import sys
#sys.exit()

if not os.path.exists(f"metrics_{prefix}"):
    try:
        os.mkdir(f"pickles_{prefix}")
        os.mkdir(f"plots_{prefix}")
        os.mkdir(f"metrics_{prefix}")
    except:
        print("Couldn't make output dirs")

train_dict = {key:value for key, value in label_dict.items() if key != target}
val_dict = {key:value for key, value in label_dict.items() if key == target}
assert target not in train_dict, "ERROR: Target protein in training dictionary"

train_dataset = DistogramDataset(labels=train_dict, pkl_dir=pkl_dir, n_classes=n_classes, two_d=True, use_pae=use_pae, n_samples_per_target=n_samples_per_target, half=half_precision,sample_pkls=True,sample_pattern=sample_pattern)
#Will use all samples for validation as well, but won't randomize which ones. Will save outputs for all samples and ensemble them at the end.
val_dataset = DistogramDataset(labels=val_dict, pkl_dir=pkl_dir, n_classes=n_classes, two_d=True, one_hot=False, use_pae=use_pae, n_samples_per_target=20, half=half_precision,sample_pkls=False)

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, pin_memory=True) #, num_workers=8, prefetch_factor=16)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)#, num_workers=8, prefetch_factor=16)




class_weights = [1.0 for i in range(n_classes)]
class_weights = torch.FloatTensor(class_weights).cuda()

loss_fn = nn.BCEWithLogitsLoss(weight=class_weights, reduction="none")
optimizer = optim.Adam(model.parameters(), eps=1e-04)


checkpoint_dir = f"checkpoints_{prefix}"
start_epoch = maybe_resume(checkpoint_dir, target, model, optimizer=optimizer, device=device)

if start_epoch >=epochs: #
    print(f"Checkpoint for epoch {start_epoch} already exists, which is >= total epochs {epochs}. Exiting.")
    sys.exit()
auroc = AUROC(task="binary")
roc = ROC(task="binary")
pc_curve = PrecisionRecallCurve(task="binary")
confmat = ConfusionMatrix(task="multiclass", num_classes=n_classes, normalize="true").to(device)

val_metrics=[]
metrics_csv=f"metrics_{prefix}/{target}.csv"
if os.path.exists(metrics_csv):
    val_metrics = pd.read_csv(metrics_csv, dtype={'target': str}).to_dict(orient="records")
    print(f"Resuming with {len(val_metrics)} validation metric entries already saved for target {target}")
for e in range(start_epoch, epochs+1):
    train_acc = 0
    train_loss = 0
    model.train()
    for i, data in enumerate(train_dataloader):

        inputs, labels, mask, pkl_file = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        mask = mask.cuda()
        optimizer.zero_grad()

        outputs = model(inputs)

        loss = loss_fn(outputs, labels) * mask[..., None]
        loss = torch.mean(loss)

        loss.backward()
        optimizer.step()

        predicted = torch.argmax(outputs, axis=-1)
        correct = (predicted == torch.argmax(labels, axis=-1)).sum().item()
        train_acc += correct / labels.size(1) ** 2
        train_loss += loss
        #train_loss += loss.item() CHECK BEFORE UNCOMMENT . This is potentially more memory efficient since it doesn't keep the computation graph,
    print(f"{datetime.datetime.now()}: Epoch {e} training accuracy: {train_acc/(i+1)}, training loss: {train_loss/(i+1)}")

    if e % 10 == 0:
        avg_loss = train_loss/(i+1)
        save_checkpoint(checkpoint_dir, model, optimizer, e, avg_loss, prefix, target)
        keep_last_n_checkpoints(checkpoint_dir, target, n=50)
        auroc.reset()
        pc_curve.reset()
        roc.reset()
        confmat.reset()
        model.eval()
        with torch.no_grad():
            #for ensemble in range(ensembles):
            #ensemble over all distograms i val_dataset for the target, which are all the samples in val_dataloader since it only contains the target protein
            data=[]
            for ensemble, val_data in enumerate(val_dataloader):
                val_inputs, val_labels, val_mask,val_pkl = val_data

                
                val_inputs = val_inputs.cuda()
                val_labels = val_labels.int().cuda()
                val_mask = val_mask.cuda()

                
                row = {}
                row["target"]=target
                row["epoch"]=e
                row["pkl"]=val_pkl[0]    
                val_output=model(val_inputs)
                
                if ensemble == 0:
                    val_outputs = val_output #model(val_inputs) # shape: 1, npairs, 3
                   
                else:
                    val_outputs += val_output
                if e==0: #only save the inputs for the first epoch since they don't change and take up a lot of space
                    row['val_input']=np.squeeze(val_inputs.cpu())
                row['val_output']=np.squeeze(val_output.cpu())
                #row['val_labels']=np.squeeze(val_labels.cpu())
                data.append(row)

            val_outputs /= (ensemble + 1)
            pickle.dump({"out":np.squeeze(val_outputs.cpu()), 
                         "labels":np.squeeze(val_labels.cpu())}, 
                         open(f"pickles_{prefix}/{target}_{e}.pkl", "wb"))

            pickle.dump(data, open(f"pickles_{prefix}/{target}_{e}_all.pkl", "wb"))

            # 2d plotting
            valid_ix = np.where(val_labels[0].cpu() >= 0)
            valid_ix = np.ix_(np.unique(valid_ix[0]), np.unique(valid_ix[1]))
            val_outputs = val_outputs[0, ...][valid_ix]
            val_labels = val_labels[0][valid_ix]

            plot_labels = val_labels.cpu().numpy()

            top_percentile_preds = val_outputs[..., n_classes-1].cpu() > np.percentile(val_outputs[..., n_classes-1].cpu(), 99)
            top_percentile_preds = np.clip(top_percentile_preds + np.transpose(top_percentile_preds), a_min=0.0, a_max=1.0) * (n_classes-1)

            plot_labels[np.tril_indices(plot_labels.shape[0])] = top_percentile_preds[np.tril_indices(plot_labels.shape[0])]

            fig, axs = plt.subplots(1,3, figsize=(18, 5))

            pred_labels = torch.argmax(val_outputs, -1).view(-1)

            cm = confmat(pred_labels, torch.clip(val_labels, 0, n_classes).view(-1))

            axs[0].imshow(plot_labels)
            axs[1].imshow(torch.argmax(val_outputs, dim=-1).cpu()/(n_classes-1))
            axs[2].imshow(cm.cpu())
            plt.savefig(f"plots_{prefix}/{target}_{e}_maps.pdf") #, dpi=200)
            plt.close()

            # 1d curves, binary metrics
            pred_pos = torch.sum(val_outputs[..., threshold_class:], -1)
            pos_labels = (val_labels >= threshold_class).int()

            au = auroc(pred_pos, pos_labels)
            precision, recall, thresholds = pc_curve(pred_pos, pos_labels)
            roc_fpr, roc_tpr, roc_thresholds = roc(pred_pos, pos_labels)
            try:
                top_k_acc = np.sum(((roc_tpr.cpu()[:k] - np.insert(roc_tpr.cpu()[:k-1], 0, 0)) > 0).numpy()) / k
            except:
                top_k_acc = -1

            fig, axs = plt.subplots(1,2, figsize=(12, 5))
            pc_curve.plot(score=True,ax=axs[0])
            aupr=float(axs[0].get_legend().get_texts()[0].get_text().split("=")[-1])
          
            roc.plot(score=True,ax=axs[1])
            plt.savefig(f"plots_{prefix}/{target}_{e}.pdf")
            plt.close()

            print(f"Epoch {e} protein {target} AUC: {au},  AUPR: {aupr}, top_10 acc: {top_k_acc}")
            row={}
            row['target']=str(target)
            row['epoch']=e
            #row['val_loss']=val_loss.item()
            #row['mcc']=mcc.item()
            row['AUC']=au.item()
            row['aupr']=aupr
            #row['precision']=best_precision.item()
            #row['recall']=best_recall.item()
            #row['f1']=best_f1.item()
            row['top_10_acc']=top_k_acc
            #row['top_20_acc']=top_20_acc
            val_metrics.append(row)

            pd.DataFrame(val_metrics).to_csv(f"metrics_{prefix}/{str(target)}.csv",index=False)

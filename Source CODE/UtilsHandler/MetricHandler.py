import math
import torch
from avalanche.benchmarks.utils.avalanche_dataset import AvalancheSubset


class MetricHandler(object):

    def __init__(self):
        pass

    def compute_prec_ratio(self, strategy, ds, params):
        """
        Compute FP for department 10.
        :param model: stategy's models
        :param ds: dataset
        :param params: dictionary of parameters
        :return:
        """
        # global_anomaly_dept = 1000
        # local_anomaly_dept = 2000

        # # Remove all global and local   anomalies
        # valid_indices = torch.where(~torch.isin(ds[:][3], torch.tensor([global_anomaly_dept, local_anomaly_dept])))[0]
        # ds = AvalancheSubset(ds, indices=valid_indices)

        # Compute loss for all dataset entries
        b_size = 32
        n_batches = math.ceil(len(ds) / b_size)

        # Compute reconstruction losses for all samples in the dataset
        criterion = torch.nn.BCELoss(reduction="none")
        recon_losses = []
        for i in range(n_batches):
            start = i * b_size
            end = (i + 1) * b_size
            x = ds[start:end][0].to(strategy.device)
            pred = strategy.model(x)
            loss = criterion(pred, x)
            loss = torch.mean(loss, dim=1)
            recon_losses.append(loss)
        recon_losses = torch.cat(recon_losses, dim=0)

        # Number of all local anomalies
        # k = torch.sum(torch.isin(ds[:][3], torch.tensor(params["target_dept_ids"]))).item()
        # recon_values, indices = torch.topk(recon_losses, k=k, largest=True)

        adaptive_thresholds = []
        window_size = 10  # You can adjust the size of the local window
        for idx in range(len(ds)):
            start = max(0, idx - window_size)
            end = min(len(ds), idx + window_size)
            local_recon_losses = recon_losses[start:end]
            adaptive_threshold = torch.mean(local_recon_losses) + 3 * torch.std(local_recon_losses)
            adaptive_thresholds.append(adaptive_threshold)

        # Number of all local anomalies
        tp = 0
        fp = 0
        for idx in range(len(ds)):
            if recon_losses[idx] > adaptive_thresholds[idx]:
                if ds[idx][3].item() in params["global_anomaly_dept"] or ds[idx][3].item() in params[
                    "local_anomaly_dept"]:
                    tp += 1
                else:
                    fp += 1

        prec = tp / float(tp + fp)

        # info = {"rec_losses": recon_values.detach().cpu().numpy().tolist(),
        #         "depts": ds[indices][3].cpu().numpy().tolist()}

        return prec, {}

    def compute_rec_ratio(self, strategy, ds, params):
        """
        Computes FN for local anomalies (dep. 2000)
        :param model:
        :param ds:
        :param params: dictionary of parameters
        :return:
        """
        # global_anomaly_dept = 1000
        # local_anomaly_dept = 2000

        # # Remove all global and local   anomalies
        # valid_indices = torch.where(~torch.isin(ds[:][3], torch.tensor([global_anomaly_dept, local_anomaly_dept])))[0]
        # ds = AvalancheSubset(ds, indices=valid_indices)

        # Compute loss for all dataset entries
        b_size = 32
        n_batches = math.ceil(len(ds) / b_size)

        # Compute reconstruction losses for all samples in the dataset
        criterion = torch.nn.BCELoss(reduction="none")
        recon_losses = []
        for i in range(n_batches):
            start = i * b_size
            end = (i + 1) * b_size
            x = ds[start:end][0].to(strategy.device)
            pred = strategy.model(x)
            loss = criterion(pred, x)
            loss = torch.mean(loss, dim=1)
            recon_losses.append(loss)
        recon_losses = torch.cat(recon_losses, dim=0)

        # Number of all local anomalies
        # k = torch.sum(torch.isin(ds[:][3], torch.tensor(params["target_dept_ids"]))).item()
        # recon_values, indices = torch.topk(recon_losses, k=k, largest=True)

        adaptive_thresholds = []
        window_size = 10  # You can adjust the size of the local window
        for idx in range(len(ds)):
            start = max(0, idx - window_size)
            end = min(len(ds), idx + window_size)
            local_recon_losses = recon_losses[start:end]
            adaptive_threshold = torch.mean(local_recon_losses) + 3 * torch.std(local_recon_losses)
            adaptive_thresholds.append(adaptive_threshold)

        # Number of all local anomalies
        tp = 0
        fn = 0
        for idx in range(len(ds)):
            if ds[idx][3].item() in params["global_anomaly_dept"] or ds[idx][3].item() in params[
                    "local_anomaly_dept"]:
                if recon_losses[idx] > adaptive_thresholds[idx]:
                    tp += 1
                else:
                    fn += 1

        rec = tp / float(tp + fn)

        return rec, {}

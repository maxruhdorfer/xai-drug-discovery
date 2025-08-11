# Training pipeline inspired by https://github.com/KRLGroup/GraphCW/tree/master

import torch
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import matplotlib.pyplot as plt
import os
from IPython import display
import numpy as np
import shutil
from torch.optim.lr_scheduler import StepLR

class Trainer(object):
    """
    A training, validation, and testing pipeline for PyTorch models
    with early stopping, learning rate scheduling, and model saving.
    """

    def __init__(self, model, train_loader, val_loader, test_loader, 
                 device, save_dir='model_save', save_name='model'):
        """
        Initialize the trainer with model, data loaders, and configurations.

        Args:
            model (torch.nn.Module): The model to train.
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            test_loader (DataLoader or None): DataLoader for test data.
            device (torch.device): Device to train on (CPU or GPU).
            save_dir (str): Directory to save models.
            save_name (str): File name prefix for saved models.
        """

        self.model = model
        self.train_loader = train_loader  
        self.val_loader = val_loader
        if test_loader != None:
            self.test_loader = test_loader
        self.device = device

        self.optimizer = None
        self.save = save_dir is not None
        self.save_dir = save_dir
        self.save_name = save_name
        check_dir(self.save_dir)

    def __loss__(self, logits, labels):
        """Compute cross-entropy loss."""
        loss = torch.nn.CrossEntropyLoss()
        return loss(logits, labels)

    def _train_batch(self, data, labels):
        """
        Train the model on a single batch.

        Args:
            data (torch_geometric.data.Data): Input batch.
            labels (torch.Tensor): Ground truth labels.

        Returns:
            tuple: (loss value, predicted classes, raw logits)
        """
        logits = self.model(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        loss = self.__loss__(logits, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), logits.argmax(-1), logits

    def _eval_batch(self, data, labels):
        """
        Evaluate the model on a single batch without backprop.

        Args:
            data (torch_geometric.data.Data): Input batch.
            labels (torch.Tensor): Ground truth labels.

        Returns:
            tuple: (loss value, predicted classes, raw logits)
        """
        self.model.eval()
        logits = self.model(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        loss = self.__loss__(logits, labels)
        loss = loss.item()
        preds = logits.argmax(-1)
        return loss, preds, logits

    def eval_val_data(self):
        """
        Evaluate model performance on the validation set.

        Returns:
            tuple: (loss, accuracy, AUC, balanced accuracy)
        """

        self.model.to(self.device)
        self.model.eval()
        losses, accs = [], []
        labels, pred_probs = [], []
        for i, batch in enumerate(self.val_loader):
            batch = batch.to(self.device)
            batch.y = batch.y.float()
            batch.y = batch.y.view(-1,2)
            batch.y = torch.argmax(batch.y, 1)
            loss, batch_preds, logits = self._eval_batch(batch, batch.y)
            accs.append(batch_preds == batch.y)
            losses.append(loss)
            probs = torch.softmax(logits, 1)
            if i == 0:
                labels = torch.tensor([[1,0] if yi==0 else [0,1] for yi in batch.y], dtype=torch.float)
                pred_probs = probs.cpu().detach()
                preds = batch_preds.cpu().detach()
            else:
                labels = torch.cat([labels, torch.tensor([[1,0] if yi==0 else [0,1] for yi in batch.y], dtype=torch.float)])
                pred_probs = torch.cat([pred_probs, probs.cpu().detach()])
                preds = torch.cat([preds, batch_preds.cpu().detach()])
        eval_loss = torch.tensor(losses).mean().item()
        eval_acc = torch.cat(accs, dim=-1).float().mean().item()
        eval_auc = roc_auc_score(labels, pred_probs)
        eval_bal_acc = balanced_accuracy_score(torch.argmax(labels,1).numpy(), preds.numpy())

        return eval_loss, eval_acc, eval_auc, eval_bal_acc

    def test(self):
        """
        Evaluate the model on the test set using the best saved checkpoint.
        """

        state_dict = torch.load(os.path.join(self.save_dir, f'{self.save_name}_best.pth'))['net']
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()
        losses, accs, pred_probs = [], [], []
        for i, batch in enumerate(self.test_loader):
            if batch.batch.shape[0] == 1:
                continue
                    
            batch = batch.to(self.device)
            batch.y = batch.y.float()
            batch.y = batch.y.view(-1,2)
            batch.y = torch.argmax(batch.y, 1)
            loss, batch_preds, logits = self._eval_batch(batch, batch.y)
            accs.append(batch_preds == batch.y)
            losses.append(loss)
            probs = torch.softmax(logits, 1)
            probs = probs[:, 1]
            if i == 0:
                labels = batch.y.cpu()
                pred_probs = probs.cpu().detach()
                preds = batch_preds.cpu().detach()
            else:
                labels = torch.cat([labels, batch.y.cpu()])
                pred_probs = torch.cat([pred_probs, probs.cpu().detach()])
                preds = torch.cat([preds, batch_preds.cpu().detach()])
        test_loss = torch.tensor(losses).mean().item()
        test_acc = torch.cat(accs, dim=-1).float().mean().item()
        test_auc = roc_auc_score(labels, pred_probs)
        test_bal_acc = balanced_accuracy_score(labels.numpy(), preds.numpy())

        print(f"Test loss: {test_loss:.4f}, test acc {test_acc:.4f}, test_auc {test_auc:.4f}, test_bal_acc {test_bal_acc:.4f}")
        return test_loss, test_acc, test_auc, test_bal_acc ,preds, pred_probs, labels

    def train(self, train_params=None, optimizer_params=None, verbose=True):
        """
        Train the model with optional early stopping and LR scheduling.

        Args:
            train_params (dict): Training parameters (num_epochs, etc.).
            optimizer_params (dict): Optimizer hyperparameters.
            verbose (bool): Whether to print progress and plot losses.

        Returns:
            tuple: Best (accuracy, AUC, balanced accuracy) from validation.
        """

        num_epochs = train_params['num_epochs']
        num_early_stop = train_params['num_early_stop']
        scheduler_step = train_params['step']
        gamma = train_params['gamma']

        # Set optimizer
        if optimizer_params is None:
            self.optimizer = Adam(self.model.parameters()) 
        else:
            self.optimizer = Adam(self.model.parameters(), **optimizer_params) 
        
        # Learning rate scheduler
        if scheduler_step is not None and gamma is not None:
            lr_schedule = StepLR(self.optimizer,
                                 step_size=scheduler_step,
                                 gamma=gamma)
        else:
            lr_schedule = None

        self.model.to(self.device)

        # Tracking best performance
        best_eval_acc = 0.0
        best_eval_auc = 0.0
        best_eval_bal_acc = 0.0
        best_eval_loss = 10000.0
        early_stop_counter = 0
        
        if verbose:
            plot = LossPlot()
        
        for epoch in range(num_epochs):
            is_best = False
            self.model.train()
            losses, accs = [], []

            for i, batch in enumerate(self.train_loader):
                batch = batch.to(self.device)
                batch.y = batch.y.float()
                batch.y = batch.y.view(-1,2)
                batch.y = torch.argmax(batch.y, 1)

                loss, batch_preds, logits = self._train_batch(batch, batch.y)
                accs.append(batch_preds == batch.y)
                losses.append(loss)
                probs = torch.softmax(logits, 1)
                probs = probs[:, 1]

                if i == 0:
                    labels = batch.y.cpu()
                    pred_probs = probs.cpu().detach()
                    preds = batch_preds.cpu().detach()
                else:
                    labels = torch.cat([labels, batch.y.cpu()])
                    pred_probs = torch.cat([pred_probs, probs.cpu().detach()])
                    preds = torch.cat([preds, batch_preds.cpu().detach()])

            train_loss = torch.FloatTensor(losses).mean().item()
            train_acc = torch.cat(accs, dim=-1).float().mean().item()
            train_auc = roc_auc_score(labels, pred_probs)
            train_bal_acc = balanced_accuracy_score(labels.numpy(), preds.numpy())
            eval_loss, eval_acc, eval_auc, eval_bal_acc = self.eval_val_data()

            if verbose:
                plot.UpdatePlots(epoch, train_loss, eval_loss, train_acc, eval_acc, train_auc, eval_auc)
            if epoch % 20 ==0 and verbose:
                print(f'Epoch:{epoch}, Training_loss:{train_loss:.4f}, Training_acc:{train_acc:.4f}, Training_auc:{train_auc:.4f}, Training_bal_acc:{train_bal_acc:.4f}, Eval_loss:{eval_loss:.4f}, Eval_acc:{eval_acc:.4f}, Eval_auc:{eval_auc:.4f}, Eval_bal_acc:{eval_bal_acc:.4f}')
            
            # Early stopping check
            if num_early_stop > 0:
                if eval_loss <= best_eval_loss:
                    best_eval_loss = eval_loss
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                if epoch > (num_epochs / 2) and early_stop_counter > num_early_stop:
                    break
            if lr_schedule:
                lr_schedule.step()

            if best_eval_bal_acc < eval_bal_acc:
                is_best = True
                best_eval_bal_acc = eval_bal_acc
                best_eval_acc = eval_acc
                best_eval_auc = eval_auc
            recording = {'epoch': epoch, 'is_best': str(is_best)}
            
            if self.save:
                self.save_model(is_best, recording=recording)
        
        print(f'Best Model: Eval_acc:{best_eval_acc:.4f}, Eval_auc:{best_eval_auc:.4f}, Eval_bal_acc:{best_eval_bal_acc:.4f}')

        return best_eval_acc, best_eval_auc, best_eval_bal_acc

    def save_model(self, is_best=False, recording=None):
        """Save model weights to file."""
        self.model.to('cpu')
        state = {'net': self.model.state_dict()}
        for key, value in recording.items():
            state[key] = value
        latest_pth_name = f"{self.save_name}_latest.pth"
        best_pth_name = f'{self.save_name}_best.pth'
        ckpt_path = os.path.join(self.save_dir, latest_pth_name)
        torch.save(state, ckpt_path)
        if is_best:
            print('saving best...')
            shutil.copy(ckpt_path, os.path.join(self.save_dir, best_pth_name))
        self.model.to(self.device)

    def load_model(self):
        """Load the best saved model weights."""
        state_dict = torch.load(os.path.join(self.save_dir, f"{self.save_name}_best.pth"))['net']
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)

    def adjust_learning_rate(self, epoch, lr):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        new_lr = lr * (0.1 ** (epoch // 10))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

class LossPlot():
    """Class for plotting training and validation loss."""

    def __init__(self):
        """Initialize the LossPlot class and set up the plot."""
        super(LossPlot, self).__init__()

        self.figLoss, self.axLoss  = plt.subplots()
        self.axLoss.set_yscale('log')

        # Display the plot dynamically in Jupyter
        self.hdisplay = display.display("", display_id = True)
        
        # Initialize empty plots for loss
        self.axLoss.plot([],[])
        self.axLoss.plot([],[])
        self.axLoss.plot([],[])
        self.axLoss.plot([],[])
        self.axLoss.plot([],[])
        self.axLoss.plot([],[])
        self.axLoss.set_xlabel('Epoch')
        self.axLoss.set_ylabel('Loss')
    
    def UpdatePlots(self, epoch, loss, validation, tacc, vacc, tauc, vauc):
        """Update the loss plots with new data.

        Parameters:
            epoch (int): Current epoch number.
            loss (float): Training loss.
            validation (float): Validation loss.
        """
        currentplot = self.axLoss.get_lines()
        currentplotLossx = currentplot[0].get_xdata()
        currentplotLossy = currentplot[0].get_ydata()
        currentplotVLossx = currentplot[1].get_xdata()
        currentplotVLossy = currentplot[1].get_ydata()
        currentplotTAccx = currentplot[2].get_xdata()
        currentplotTAccy = currentplot[2].get_ydata()
        currentplotVAccx = currentplot[3].get_xdata()
        currentplotVAccy = currentplot[3].get_ydata()
        currentplotTAucx = currentplot[4].get_xdata()
        currentplotTAucy = currentplot[4].get_ydata()
        currentplotVAucx = currentplot[5].get_xdata()
        currentplotVAucy = currentplot[5].get_ydata()

        # Remove old plots and add new data
        currentplot[1].remove()
        currentplot[0].remove()
        currentplot[2].remove()
        currentplot[3].remove()
        currentplot[4].remove()
        currentplot[5].remove()
        self.axLoss.plot(np.append(currentplotLossx,int(epoch)),
                        np.append(currentplotLossy, loss),'b-', label='Training Loss')

        self.axLoss.plot(np.append(currentplotVLossx,int(epoch)),
                        np.append(currentplotVLossy, validation),'r-', label='Validation Loss')

        self.axLoss.plot(np.append(currentplotTAccx,int(epoch)),
                        np.append(currentplotTAccy, tacc),'g--', label='Training Accuracy')

        self.axLoss.plot(np.append(currentplotVAccx,int(epoch)),
                        np.append(currentplotVAccy, vacc),'g-', label='Validation Accuracy')

        self.axLoss.plot(np.append(currentplotTAucx,int(epoch)),
                        np.append(currentplotTAucy, tauc),'y--', label='Training AUC')

        self.axLoss.plot(np.append(currentplotVAucx,int(epoch)),
                        np.append(currentplotVAucy, vauc),'y-', label='Validation AUC')

        self.axLoss.set_xlabel('Epoch')
        self.axLoss.set_ylabel('Loss')
        self.axLoss.legend()
        
        # Update the display
        self.hdisplay.update(self.figLoss)

def check_dir(save_dirs):
    """Create a directory if it does not exist."""
    if save_dirs:
        if os.path.isdir(save_dirs):
            pass
        else:
            os.makedirs(save_dirs)
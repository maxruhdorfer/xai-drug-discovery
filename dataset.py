from torch_geometric.data import InMemoryDataset, Data
import pandas as pd
from rdkit.Chem import AllChem
import torch
import numpy as np
import random

# Mapping of bond types to integer encodings
BOND_TYPE = {
    "SINGLE": 0,
    "DOUBLE": 1,
    "TRIPLE": 2,
    "AROMATIC": 3
}

# List of classification tasks in the Tox21 dataset
TASK_LIST = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
                 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']

# Possible atomic numbers for atoms in dataset molecules
ATOMIC_NUMBERS = [1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 19, 20, 22, 23, 24, 25, 
                  26, 27, 28, 29, 30, 32, 33, 34, 35, 38, 40, 42, 46, 47, 48, 49, 50, 51, 
                  53, 56, 60, 64, 66, 70, 78, 79, 80, 81, 82, 83]

# Possible atom degrees (number of directly bonded neighbors)
ATOMIC_DEGREE = [0, 1, 2, 3, 4, 5, 6]

# Possible formal charges for atoms
ATOMIC_FORMAL_CHARGE = [-2, -1, 0, 1, 2, 3]

class Tox21Dataset(InMemoryDataset):
    """
    PyTorch Geometric InMemoryDataset for the Tox21 molecular dataset.
    """

    def __init__(self, root, task=TASK_LIST, transform=None, pre_transform=None):
        """
        Args:
            root (str): Root directory where the dataset is stored.
            task (list): List of tasks to use for classification.
            transform (callable, optional): Transform applied on each data sample.
            pre_transform (callable, optional): Transform applied before saving data.
        """
        self.tasks = task
        super().__init__(root, transform, pre_transform)

        # Load preprocessed dataset
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        """Name of raw data file expected in raw_dir."""
        return 'tox21.csv'

    @property
    def processed_file_names(self):
        """Name of processed dataset file."""
        return 'tox21.pt'

    def download(self):
        """Dataset must already exist locally; no download provided."""
        raise NotImplementedError('Class assumes tox21.csv is in same directory. '
                                  'No download allowed')

    def process(self):
        """
        Processes the raw dataset file into a list of Data objects and saves it.
        """
        data_list = []

        rdkit_mol_objs, labels = \
                self._load_tox21_dataset(self.raw_paths[0])
        
        for i in range(len(rdkit_mol_objs)):
            rdkit_mol = rdkit_mol_objs[i]
            if rdkit_mol is None:
                continue

            data = self._mol_to_graph(rdkit_mol)
            
            data.id = torch.tensor([i])  # Store molecule index
            # the dataset
            if len(labels[i,:]) == 1:
                # Binary classification handling for single-task case
                if labels[i,0] == 0:
                    continue
                elif labels[i,0] == -1:
                    data.y = torch.tensor([1,0], dtype=torch.float)
                else:
                    data.y = torch.tensor([0,1], dtype=torch.float)
            else:
                # Multi-task case
                data.y = torch.tensor(labels[i, :])
            data_list.append(data)

        # Save processed dataset
        self.save(data_list, self.processed_paths[0])

    def _load_tox21_dataset(self, input_path):
        """
        Loads the Tox21 dataset from a CSV file.

        Args:
            input_path (str): Path to the raw CSV file.

        Returns:
            tuple: (list of RDKit Mol objects, numpy array of labels)
        """
        input_df = pd.read_csv(input_path, sep=',')
        smiles_list = input_df['smiles']

        # Convert SMILES strings to RDKit molecule objects
        rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
        labels = input_df[self.tasks]

        labels = labels.replace(0, -1) # Convert 0 to -1 for inactive

        labels = labels.fillna(0)# Replace NaN with 0

        assert len(smiles_list) == len(rdkit_mol_objs_list)
        assert len(smiles_list) == len(labels)

        return rdkit_mol_objs_list, labels.values

    def _mol_to_graph(self, mol):
        """
        Converts an RDKit molecule to a PyG Data graph.

        Args:
            mol (rdkit.Chem.Mol): RDKit molecule object.

        Returns:
            torch_geometric.data.Data: Graph representation of molecule.
        """
        # atoms
        atom_features_list = []
        edge_features_list = []
        edge_indices_list = []
        
        # Encode atom features
        for a in mol.GetAtoms():
            one_hot_atom_num = [0]*len(ATOMIC_NUMBERS)
            one_hot_atom_deg = [0]*len(ATOMIC_DEGREE)
            one_hot_atom_charge = [0]*len(ATOMIC_FORMAL_CHARGE)
            idx_num = ATOMIC_NUMBERS.index(a.GetAtomicNum())
            if idx_num >= 0:
                one_hot_atom_num[idx_num] =1
            idx_deg = ATOMIC_DEGREE.index(a.GetDegree())
            if idx_deg >= 0:
                one_hot_atom_deg[idx_deg] =1
            idx_charge = ATOMIC_FORMAL_CHARGE.index(a.GetFormalCharge())
            if idx_charge >= 0:
                one_hot_atom_charge[idx_charge] =1
            atom_features_list.append(one_hot_atom_num + one_hot_atom_deg + one_hot_atom_charge + [
                int(a.IsInRing()),
                1-int(a.IsInRing()),
                int(a.GetIsAromatic()),
                1-int(a.GetIsAromatic())
            ])
            
        x = torch.tensor(np.array(atom_features_list), dtype=torch.float)
    
        # Encode bond features
        for b in mol.GetBonds():
            one_hot_type = [0] * 4
            idx = BOND_TYPE.get(str(b.GetBondType()).upper(), -1)
            if idx >= 0:
                one_hot_type[idx] = 1
            edge_features = one_hot_type + [
                int(b.GetIsAromatic()),
                1-int(b.GetIsAromatic()),
                int(b.GetIsConjugated()),
                1-int(b.GetIsConjugated()),
                int(b.IsInRing()),
                1-int(b.IsInRing())
            ]

            # Add edges in both directions for undirected graphs
            edge_features_list.append(edge_features)
            edge_features_list.append(edge_features)
            edge_indices_list.append((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
            edge_indices_list.append((b.GetEndAtomIdx(), b.GetBeginAtomIdx()))
    
        edge_index = torch.tensor(np.array(edge_indices_list).T, dtype=torch.long)
    
        edge_attr = torch.tensor(np.array(edge_features_list),
                                     dtype=torch.float)

    
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, mol=mol)
    
        return data

def random_split_dataset(dataset, frac_train=0.8, frac_val=0.1, frac_test=0.1):
    """
    Randomly splits dataset into train, validation, and test subsets.

    Args:
        dataset (Dataset): PyG dataset to split.
        frac_train (float): Fraction of samples for training.
        frac_val (float): Fraction of samples for validation.
        frac_test (float): Fraction of samples for testing.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """

    num_mols = len(dataset)
    all_idx = list(range(num_mols))
    random.shuffle(all_idx)

    train_idx = all_idx[:int(frac_train * num_mols)]
    val_idx = all_idx[int(frac_train * num_mols):int(frac_val * num_mols)
                                                   + int(frac_train * num_mols)]
    test_idx = all_idx[int(frac_val * num_mols) + int(frac_train * num_mols):]

    assert len(train_idx) + len(val_idx) + len(test_idx) == num_mols

    train_dataset = dataset[torch.tensor(train_idx)] if len(train_idx) > 0 else Data()
    val_dataset = dataset[torch.tensor(val_idx)] if len(val_idx) > 0 else Data()
    test_dataset = dataset[torch.tensor(test_idx)] if len(test_idx) > 0 else Data()

    return train_dataset, val_dataset, test_dataset
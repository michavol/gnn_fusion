from dgl.data import ZINCDataset
import pickle

train = ZINCDataset(mode='train')
val = ZINCDataset(mode='valid')
test = ZINCDataset(mode='test')
num_atom_type = train.num_atom_types
num_bond_type = train.num_bond_types

with open('molecules/ZINC.pkl','wb') as f:
    pickle.dump([train,val,test,num_atom_type,num_bond_type],f)


class OTTO():
    def __init__(self, debug=False):
        #load modules and dir_path
        self.data = self.load_otto(debug)
        
    def load_otto(self, debug):   
        import glob
        from collections import Counter
        import numpy as np
        import pandas as pd     
        data_folder = '/Users/linshoahchieh/Documents/Kaggle/Otto_recommender_system/data/'
        
        #load train
        train_folder = data_folder + 'train/*'
        dfs = list()
        for p in glob.glob(train_folder):
            dfs.append(pd.read_parquet(p))

        train = pd.concat(dfs)
        
        # reduce train if debugging
        if debug:
            print('Origian train size: {}'.format(train.shape))
            sampled_sized = 10000
            samepled_indices = np.random.choice(train['session'].unique(), sampled_sized)
            train = train.loc[train['session'].isin(samepled_indices)]
            print('Train size after reduction: {}'.format(train.shape))

        #load test
        test_folder = data_folder + 'test/*'
        dfs = list()
        for p in glob.glob(train_folder):
            dfs.append(pd.read_parquet(p))

        test = pd.concat(dfs)

        #load val
        val_folder = data_folder + 'validation/'
        val = pd.read_parquet(val_folder + 'val.parquet')
        val_true = pd.read_parquet(val_folder + 'val_true.parquet')

        #reduce val if debugging
        if debug:
            print('Original val size: {}'.format(val.shape))
            sampled_sized = 10000
            samepled_indices = np.random.choice(val['session'].unique(), sampled_sized)
            val = val.loc[val['session'].isin(samepled_indices)]
            val_true = val_true.loc[val_true['session'].isin(samepled_indices)]
            print('Val size after reduction: {}'.format(val.shape))

        #generate candidates
        candidates = Counter(train['aid']).most_common(20)
        candidates = [aid for aid,counts in candidates]

        #into a dict
        data = {
            'train': train,
            'test': test,
            'val': val,
            'val_true': val_true,
            'candidates': candidates,
            'types': ['clicks', 'carts', 'orders']
        }

        return data

    def recall_singel_row(self, preds, targets):
        preds = preds.split(' ')
        targets = targets.split(' ')
        intersection = len([t for t in targets if t in preds])
        return intersection,len(targets)

    def from_sub_to_recall20(self, sub):
        from tqdm import tqdm
        assert sub.shape[1] == 2, 'Subs shape has to be (x,2)'
        assert type(sub['labels'][0]) == str, 'Labels has to be strings'

        # Convert sub to dict
        sub_mapping = sub.set_index('session_type').to_dict()['labels']

        # Calculation
        print('Calculating recall20 ...')
        types = self.data['types']
        val_true = self.data['val_true']
        sum_intersection,sum_n_targets = {t:0 for t in types},{t:0 for t in types}

        # Iterrows and calculate recall
        for i,row in tqdm(val_true.iterrows()):
            t = row['type']
            key = str(row['session']) + '_' + t
            preds = sub_mapping[key]
            targets = row['target']
            (intersection, n_target) = self.recall_singel_row(preds, targets)
            sum_intersection[t] += intersection
            sum_n_targets[t] += n_target

        # Weight averaing recall scores
        weights = [0.1, 0.3, 0.6]
        recalls = [sum_intersection[t]/sum_n_targets[t] for t in types]
        recall = 0
        for i in range(3):
            recall += weights[i] * recalls[i]

        print('Final recall score: {:.6f}'.format(recall))

    def baseline(self):
        import pandas as pd
        print('Using candidates, themost frequent items, as predictions ...')
        # Associate session indices with candidates
        candidates_in_str = ' '.join([str(c) for c in self.data['candidates']])
        preds = self.data['val'].groupby('session').apply(lambda x: candidates_in_str)

        # Convert association data (pd series) to submission formating 
        ls_suffix = ['_clicks', '_carts', '_orders']
        ls_dfs = list()
        for suf in ls_suffix:
            ls_dfs.append(pd.DataFrame(preds.add_suffix(suf), columns=['labels']).reset_index())

        # Combine predictions for each type
        mapping = {'session': 'session_type'}
        sub = pd.concat(ls_dfs).sort_values('session').rename(columns=mapping).reset_index(drop=True)

        # Convert sub to dict
        self.from_sub_to_recall20(sub)
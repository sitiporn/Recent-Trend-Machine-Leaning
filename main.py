from config import config as config
from module import * 
from sklearn.model_selection import KFold
 

data = load_data()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

kfold = KFold(n_splits=config['k_folds'], shuffle=True)
if(config["Do print device"]):
    print(device)

best_acc_ = 0.0
bes_hyperparmeter_index = 0

for i in range(config['number_of_optimizer']):
    sum_acc_fold = 0.0
    for fold, (train_ids, test_ids) in enumerate(kfold.split(data)):
        #print(fold, train_ids, test_ids)
        # Print
        print(f'FOLD {fold}')
        print('--------------------------------')
        
        
        ressenet2 = ResSENet18().to(device)
        ressenet2.load_state_dict(torch.load('rtml/ressenet18_bestsofar_3.pth'))
        ressenet2.linear =  nn.Linear(in_features=512, out_features=2, bias=True).to(device)
        
        
        #print(ressenet2.eval())
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        
        # Define data loaders for training and testing data in this fold
        train_dataloader = torch.utils.data.DataLoader(data, 
                        batch_size=7, sampler=train_subsampler)
        val_dataloader = torch.utils.data.DataLoader(
                        data,
                        batch_size=2, sampler=test_subsampler)
        criterion2 = nn.CrossEntropyLoss()
        params_to_update2 = ressenet2.parameters()
        optimizer2 = optim.Adam(params_to_update2, lr= config['learning_rate'][str(i)])

        #name = 'ressenet18_bestsofar'+str(fold)
       # print(name)
        #print(ressenet2.eval()) 
        
        dataloaders = {'train': train_dataloader, 'val': val_dataloader}
        _ , val_acc_history2, loss_acc_history2, best_acc_current = train_model(ressenet2, dataloaders, criterion2, optimizer2,config['Epochs'],'ressenet18_bestsofar_4')
        
        sum_acc_fold += best_acc_current
        
        print('--------------------------------')
    # print(best_acc)
    # print((sum_acc_fold*1.0)/8)
    print("Model"+str(i),"avg_acc:",sum_acc_fold/config['k_folds'])
    if best_acc_current > best_acc_:
        best_acc_ = best_acc_current
        bes_hyperparmeter_index = i


## Test Model
test_model()

print("The best hyperparameter is : Model"+str(bes_hyperparmeter_index))



# here you write script for model training, e.g. loading data, initializing model and trainer, etc.
# save results, checkpoints, etc. in the outputs directory, but dont push model weights to git, instead save them locally and add them to .gitignore
from training.trainer import TrainerArgs, Trainer
from torch.nn import MSELoss, L1Loss
from torch.optim import Adam
from models.regression import SimpleRegressionNet
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from pandas import read_csv
from datasets.regression import RegressionTableDataset
from torch.utils.data import DataLoader
import kagglehub
import torch
import pandas as pd

def r2(predictions, targets):
    return r2_score(
        targets.detach().cpu().numpy().ravel(),
        predictions.detach().cpu().numpy().ravel(),
    )


def train(model, args, optimizer, train_data_loader, valid_data_loader):
    trainer = Trainer(model, args, optimizer, train_data_loader, valid_data_loader)
    trainer.train()

if __name__ == "__main__":
    

    loss = MSELoss()
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'

    args = TrainerArgs(
        loss,
        epochs=100,
        device=device,
        checkpoint_step=10,
        metrics={
            "R2": r2
        },
    )
    path = kagglehub.dataset_download("mirichoi0218/insurance")
    data = read_csv(f'{path}/insurance.csv')
    s = (data.dtypes == 'str')
    obj = list(s[s].index)
    oh_encoder = OneHotEncoder(sparse_output=False)
    ohe_data = oh_encoder.fit_transform(data[obj])
    encoded_columns_name = oh_encoder.get_feature_names_out(obj)
    data = data.drop(columns=obj)
    data = pd.concat([data, pd.DataFrame(ohe_data, columns=encoded_columns_name)], axis=1)
    X, y = data.drop(columns=["charges"]), data["charges"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=42)

    x_scaler = StandardScaler()
    X_train = pd.DataFrame(x_scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_val = pd.DataFrame(x_scaler.transform(X_val), columns=X_val.columns, index=X_val.index)

    y_scaler = StandardScaler()
    y_train = pd.Series(y_scaler.fit_transform(y_train.to_frame()).ravel(), index=y_train.index)
    y_val = pd.Series(y_scaler.transform(y_val.to_frame()).ravel(), index=y_val.index)
    train_dataset = RegressionTableDataset(X_train, y_train)
    valid_dataset = RegressionTableDataset(X_val, y_val)
    train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_data_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
    
    model = SimpleRegressionNet(input_dim=X.shape[1])
    optimizer = Adam(model.parameters(), args.learning_rate)
    
    train(model, args, optimizer, train_data_loader, valid_data_loader)
    
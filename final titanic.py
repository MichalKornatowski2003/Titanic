import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

class Data_manipulation:

    def dataDoing(self):
        """

        :return: Creates Data Frame which is going to be used to train our model
         to learn as much about Titanic disaster as possible
        """
        df = pd.read_csv('../PyTorch/PyTorch2/venv/tested.csv', delimiter=',', index_col='PassengerId')
        return df

    def checkAge(age):
        if age < 18:
            return 0
        elif 18 <= age < 60:
            return 1
        else:
            return 2

    def modifyData(self, df):
        """

        :param df: Dataframe which we are going to work with
        :return: Modifies Data Frame Columns erasing unnecessary ones and adding important ones
        """
        df = df.drop(columns=["Name", "Ticket", "Cabin"])
        df = df.fillna(df["Age"].mean())
        df = df.fillna(df["Fare"].mean())
        map_sex = {"female": 0, "male": 1}
        map_embarked = {"S": 0, "Q": 1, "C": 2}
        df["Sex"] = df["Sex"].map(map_sex)
        df["Embarked"] = df["Embarked"].map(map_embarked)
        df["Adulthood"] = df["Age"].apply(Data_manipulation.checkAge)
        return df

    def splits(self, df):
        """

        :param df: Still the same Data Frame
        :returns: Splits Data Frame for train part and test part in proportions of 80% to 20%
        of the Data Frame
        """
        train_part, test_part = train_test_split(df, test_size=.2, train_size=.8)
        return train_part, test_part

    def dataframeToTensor(self, train_part, test_part):
        """
        :params train_part, test_part: Data parts in Data Frame dtype
        :return: Transforms Data Frame to tensor to pass the data to Data Loader
        """
        train_set = torch.tensor(data=train_part.values, dtype=torch.float32)
        test_set = torch.tensor(data=test_part.values, dtype=torch.float32)
        return train_set, test_set

    def dataLoader(self, train_set, test_set, BATCH_SIZE):
        """

        :param train_set: set used for training
        :param test_set: set used for testing
        :param BATCH_SIZE: number of batches
        :return: Data Loaders for model to train and test it
        """
        train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=False)
        return train_loader, test_loader


data = Data_manipulation()
dataFrame = data.dataDoing()
dataFrame = data.modifyData(df=dataFrame)
df_train, df_test = data.splits(df=dataFrame)
train_set, test_set = data.dataframeToTensor(train_part=df_train, test_part=df_test)
BATCH_SIZE = 16
train_loader, test_loader = data.dataLoader(train_set=train_set, test_set=test_set, BATCH_SIZE=BATCH_SIZE)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Flatten(), nn.Linear(8, 64), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(64, 48), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Linear(48, 32), nn.ReLU())
        self.layer4 = nn.Sequential(nn.Linear(32, 24), nn.ReLU())
        self.layer5 = nn.Sequential(nn.Linear(24, 12), nn.ReLU())
        self.layer6 = nn.Sequential(nn.Linear(12, 1))

    def forward(self, x):
        """

        :param x: Data for the model to traverse all layers
        :return:
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x


class Learning:
    def accuracy_fn(self, y_true, y_pred):
        """

        :param y_true: Correct value
        :param y_pred: Predicted value
        :return: Calculates how accurate the test was 
        """
        y_pred = (y_pred > 0.5).float()
        correct = (y_true == y_pred).float().sum()  # Calculates how many predictions were correct
        accuracy = correct / len(y_true)
        return accuracy * 10

    def training(self, data_loader, model):
        """

        :param data_loader: The data to train our model
        :param model: The model to train
        :return: Loss and accuracy of our model after train
        """
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.SGD(params=model.parameters(), lr=0.001)
        epochs = 200
        accuracies = []
        losses = []
        for epoch in range(epochs):

            running_lose = 0.0
            running_accuracy = 0.0
            for i, given_data in enumerate(train_loader):
                result = given_data[:, 0]
                train_data = given_data[:, 1:]
                optimizer.zero_grad()
                pred = model.forward(train_data)
                loss = criterion(pred.squeeze(), result)
                loss.backward()
                optimizer.step()
                prediction = pred.sigmoid().round()
                running_accuracy += learning.accuracy_fn(y_true=result, y_pred=prediction)
                running_lose += loss.item()

            epoch_loss = running_lose / len(data_loader)
            epoch_accuracy = running_accuracy / len(data_loader)
            accuracies.append(epoch_accuracy)
            losses.append(epoch_loss)
            print(f"Epoch: {epoch + 1}, Accuracy: {epoch_accuracy}%, Loss: {epoch_loss}")
        return losses, accuracies

    def display_loss_and_accuracy(self, losses, accuracies):
        """

        :param losses: Loss of the model
        :param accuracies: Accuracy of the model
        :return: Show how our model is learning
        """
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(losses)
        plt.title('Loss per epoch')
        plt.subplot(1, 2, 2)
        plt.plot(accuracies)
        plt.title('Accuracy per epoch')
        plt.show()


class ModelOperations:

    def modelSave(self, model, path):
        """

        :param model: Model to save
        :param path: Path to the saved model
        :return:
        """
        torch.save(model.state_dict(), path)

    def modelLoad(self, path):
        """

        :param path: Path to the saved model
        :return: Loaded model
        """
        model1 = Model()
        model1.load_state_dict(torch.load(path))
        return model1

    def split_results_and_input(self, data_loader):
        """

        :param data_loader: Data to split for results and data for the model
        :return:
        """
        all_data = next(iter(data_loader))
        result = all_data[:, 0]
        input = all_data[:, 1:]
        return result, input

    def print_answers(self, result, prediction):
        """

        :param result: Real results what our model could produce
        :param prediction: Predictions of our model
        :return:
        """
        good_answers = 0
        for i in range(BATCH_SIZE):
            print(f"\nResult: {result[i].item()}, Prediction: {prediction[i].item()}")
            if result[i].item() == prediction[i].item():
                good_answers += 1
        print(f"\nCorrect guesses: {good_answers} / {BATCH_SIZE}\n\n\n\n")

    def print_data(self, path, train_loader):
        """

        :param path: Path to the saved model
        :param train_loader: Data to train the model
        :return:
        """
        model = self.modelLoad(path=path)
        result, train_data = self.split_results_and_input(train_loader)
        outputs = model.forward(train_data)
        prediction = outputs.sigmoid().round()
        self.print_answers(result=result, prediction=prediction)

    def test_model(self, test_loader, path):
        """

        :param test_loader: Data to test how our model works
        :param path: Path to the saved model
        :return:
        """
        result, train_data = self.split_results_and_input(test_loader)
        model = self.modelLoad(path=path)
        prediction = model(train_data).sigmoid().round()
        self.print_answers(result=result, prediction=prediction)
        return result, prediction


learning = Learning()
model = Model()

losses_train, accuracies_train = learning.training(data_loader=train_loader, model=model)
learning.display_loss_and_accuracy(losses=losses_train, accuracies=accuracies_train)

path = ".\model.pth"

modeloperations = ModelOperations()
modeloperations.modelSave(model=model, path=path)
loaded_model = modeloperations.modelLoad(path=path)
modeloperations.print_data(path=path, train_loader=train_loader)
result, prediction = modeloperations.test_model(test_loader=test_loader, path=path)

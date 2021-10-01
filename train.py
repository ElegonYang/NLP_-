from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging
import torch
import random


class MachineLearning:

    def __init__(self, model, model_name_main, batch_size, epoch):
        self.model = model
        self.model_name_main = model_name_main
        self.batch_size = batch_size
        self.epoch = epoch

    def do(self):

        self.log()
        train_data = self.import_train_data_csv_to_train_data_list('train.csv')
        df = self.train_data_list_to_dataframe(train_data)
        setting = self.train_model_set(38, 60)
        setting.train_model(df)

    @staticmethod
    def log():
        logging.basicConfig(level=logging.INFO)
        transformers_logger = logging.getLogger("transformers")
        transformers_logger.setLevel(logging.WARNING)
        print(f"是否使用 GPU:{torch.cuda.is_available()}")

    @staticmethod
    def import_train_data_csv_to_train_data_list(csv_name):

        train_data = []
        train_csv = pd.read_csv(csv_name)
        list_dataset_train = train_csv.values.tolist()
        for dataset in list_dataset_train:
            train_data.append([dataset[1], dataset[2]])

        return train_data

    @classmethod
    def train_data_list_to_dataframe(cls, train_data):
        random.shuffle(train_data)
        train_df = pd.DataFrame(train_data)
        train_df.columns = ["ID", "sentiment"]

        return train_df

    def train_model_set(self, batch_size, epoch):

        output_dir = f"outputs/{self.model_name_main}-bs-{batch_size}-ep-{epoch}-cls-model/"

        model_args = ClassificationArgs()
        model_args.train_batch_size = batch_size
        model_args.num_train_epochs = epoch
        model_args.overwrite_output_dir = True
        model_args.reprocess_input_data = True
        model_args.use_multiprocessing = True
        model_args.save_model_every_epoch = True
        model_args.output_dir = output_dir

        model = ClassificationModel(
            model_type=self.model,
            model_name=self.model_name_main,
            use_cuda=torch.cuda.is_available(),
            cuda_device=0,
            args=model_args,
            )

        return model


go = MachineLearning('roberta', 'roberta-base', 38, 60)
# def __init__(self, model, model_name_main, batch_size, epoch):

if __name__ == '__main__':
    go.do()

from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging
import time
import torch


class Predict:

    test_data = []
    list_result = []

    def __init__(self, model, model_name, batch_size, epoch):

        self.model = model
        self.model_name = model_name
        self.batch_size = batch_size
        self.epoch = epoch

    def do(self):

        self.log()
        self.predict_csv_get('test.csv')
        out_put_dir = self.predict_model_set()
        model_name = self.model_args_set(out_put_dir)
        self.predict_result(model_name)
        self.write_predict_result_to_csv(out_put_dir)

    @staticmethod
    def log():
        logging.basicConfig(level=logging.INFO)
        transformers_logger = logging.getLogger("transformers")
        transformers_logger.setLevel(logging.WARNING)
        print(f"是否使用 GPU:{torch.cuda.is_available()}")

    @classmethod
    def predict_csv_get(cls, csv_name):
        test_csv = pd.read_csv(csv_name)
        list_dataset = test_csv.values.tolist()
        for dataset in list_dataset:
            cls.test_data.append(dataset[1])

    def predict_model_set(self):
        output_dir = f"outputs/{self.model}-bs-{self.batch_size}-ep-{self.epoch}-cls-model/"
        return output_dir

    def model_args_set(self, output_dir):
        model_args = ClassificationArgs()
        model_args.train_batch_size = self.batch_size
        model_args.num_train_epochs = self.epoch
        model_args.overwrite_output_dir = True
        model_args.reprocess_input_data = True
        model_args.use_multiprocessing = True
        model_args.save_model_every_epoch = True
        model_args.output_dir = output_dir

        model = ClassificationModel(
            model_type=self.model,
            model_name=self.model_name,
            use_cuda=torch.cuda.is_available(),
            cuda_device=0,
            args=model_args
        )

        return model

    @classmethod
    def predict_result(cls, model):
        predictions, raw_outputs = model.predict(cls.test_data)

        for index, sentiment in enumerate(predictions):
            cls.list_result.append([cls.test_data[index][0], sentiment])

    def write_predict_result_to_csv(self, out_put_dir):

        result_df = pd.DataFrame(self.list_result)
        result_df.columns = ["ID", "sentiment"]
        result_df.to_csv(out_put_dir, index=False)




# def __init__(self, model, model_name, batch_size, epoch):
go = Predict('roberta', 'roberta-base', 38, 47)

if __name__ == '__main__':
    go.do()






# tensorboard
# tensorboard --logdir=D:\PycharmProjects\CodingLife\Aidea\runs\Jul25_09-06-37_DESKTOP-NEV29I7
# tensorboard --logdir=D:\PycharmProjects\CodingLife\Aidea\runs\Jul26_00-04-42_DESKTOP-NEV29I7
# tensorboard --logdir=D:\PycharmProjects\CodingLife\Aidea\runs\Jul26_14-06-30_DESKTOP-NEV29I7
# tensorboard --logdir=D:\PycharmProjects\CodingLife\BDSE20\runs\Sep09_17-56-00_DESKTOP-NEV29I7

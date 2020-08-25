import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from simpletransformers.classification import ClassificationModel
# from package_tweet.custom_classification_model import CustomClassificationModel
from sklearn import metrics
from sklearn.model_selection import KFold, StratifiedKFold
import pandas as pd
import numpy as np
import torch


# used to test if cuda support is enabled
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def main():
    # abspath resolves redundant separators and up-level references
    data_dir = os.path.abspath(os.path.join(os.getcwd(), "../data/data_base"))

    # Read the data from files
    train_df = pd.read_csv(os.path.join(data_dir, "trn_A.csv"))
    valdn_df = pd.read_csv(os.path.join(data_dir, "val_A.csv"))
    # test_df = pd.read_csv(os.path.join(data_dir, "tst_A.csv"))
    # print(train_df.shape, valdn_df.shape, test_df.shape)

    # change datatype of labels column from int to float to avoid runtime warning with respect to double_scalars (see Issue #2)
    train_df = train_df.astype({"labels_for_settingA": np.float64})
    valdn_df = valdn_df.astype({"labels_for_settingA": np.float64})

    training_args = {
        'overwrite_output_dir': True,
        'num_train_epochs': 3,
        'train_batch_size': 32,
        # 'gradient_accumulation_steps': 8,
    #     'evaluate_during_training': True,
    #     'evaluate_during_training_verbose': True
    }

    # Binary Classification using Simple Transformers
    # model = ClassificationModel('bert', 'bert-base-uncased', use_cuda=False, args=training_args)
    # model = CustomClassificationModel('bert', 'bert-large-uncased', use_cuda=True)
    # model = CustomClassificationModel('bert', 'bert-base-cased', use_cuda=True)
    # model = CustomClassificationModel('bert', 'bert-large-cased', use_cuda=True)
    # model = CustomClassificationModel('xlnet', 'xlnet-base-cased', use_cuda=True)
    # model = CustomClassificationModel('xlnet', 'xlnet-large-cased', use_cuda=True)
    # print("***************************************************************")
    # print("---------------------------------------------------------------")
    # print("MODEL TYPE: %s" % model.args["model_type"])
    # print("MODEL NAME: %s" % model.args["model_name"])
    # print(model)
    # print(model.args)
    # print("---------------------------------------------------------------")
    # print("***************************************************************")

    # train the model (fine-tune the pre-trained model on the train split)

    # using k-fold cross validation
    ''' Validated my implementation of k-fold CV using 
    https://medium.com/@schmidphilipp1995/k-fold-as-cross-validation-with-a-bert-text-classification-example-4017f76a863a '''
    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=53)
    # skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=53)

    cv_acc = list()
    cv_f1 = list()
    cv_loss = list()
    model = None

    for train_idx, test_idx in kf.split(train_df):
        cv_train_folds = train_df.iloc[train_idx]
        cv_test_fold = train_df.iloc[test_idx]

        # if first epoch, initialize a new model; otherwise reuse model saved in previous epoch
        if model is None:
            # instantiate the model
            model = ClassificationModel('bert', 'bert-base-uncased', use_cuda=False, args=training_args)
            # print("***************************************************************")
            # print("---------------------------------------------------------------")
            print("MODEL TYPE: %s" % model.args["model_type"])
            print("MODEL NAME: %s" % model.args["model_name"])
            # print(model)
            print(model.args)
            # print("---------------------------------------------------------------")
            # print("***************************************************************")
        else:
            print("Not the first epoch!")
            #model = BertForSequenceClassification.from_pretrained('./directory/to/save/')  # re-load


        # train the model on (k-1) folds
        model.train_model(cv_train_folds)
        # validate the model on k-th fold
        result, model_outputs, wrong_preds = model.eval_model(cv_test_fold, accuracy=metrics.accuracy_score,
                                                              f1=metrics.f1_score)
        # append the result to global results list
        cv_acc.append(result['accuracy'])
        cv_f1.append(result['f1'])
        # cv_loss.append(result['loss'])

        model._save_model(output_dir="outputs/saved_models/")  # save



    # print("\n\nAccuracy over %d folds: " % k)
    # print(cv_acc)
    # print("\nF1-score over %d folds:" % k)
    # print(cv_f1)
    # # print("\nLoss over %d folds:" % k)
    # # print(cv_loss)
    #
    # print("\n\n5-Fold Cross-Validation Results")
    # print(f"Average accuracy: {sum(cv_acc) / len(cv_acc)}")
    # print(f"Average f1: {sum(cv_f1) / len(cv_f1)}")
    # # print(f"Average loss: {sum(cv_loss) / len(cv_loss)}")

    # tmp1 = valdn_df[valdn_df.loc[:, 'labels_for_settingA'] == 1.0]
    # print("Optimistic tweets in validation set: ", tmp1.shape)
    # tmp2 = valdn_df[valdn_df.loc[:, 'labels_for_settingA'] == 0.0]
    # print("Pessimistic tweets in validation set: ", tmp2.shape)
    #
    # # results, model_outputs, wrong_predictions = model.eval_model(valdn_df, accuracy=sklearn.metrics.accuracy_score,
    # #                                                             f1=sklearn.metrics.f1_score)
    # # print(results)
    # # print the raw outputs from the model
    # # print(model_outputs[0:5])
    # # print the misclassifications by the model
    # # print(wrong_predictions)
    #
    #
    # # use the predict method to make predictions and then use these predictions for evaluation
    # # predict method requires examples to be formatted as a list
    # # valdn_samples = valdn_df['Tweet'].to_list()
    # # valdn_true_lbls = valdn_df['labels_for_settingA'].to_list()
    #
    # # print(valdn_df[0:5])
    # # print(valdn_samples[0:5])
    # # print(valdn_true_lbls[0:5])
    #
    # # pred_labels, model_outputs = model.predict(valdn_samples)
    # # print(len(pred_labels))
    # # acc = metrics.accuracy_score(valdn_true_lbls, pred_labels)
    # # f1 = metrics.f1_score(valdn_true_lbls, pred_labels)
    # # cm = metrics.confusion_matrix(valdn_true_lbls, pred_labels)
    # #
    # # print()
    # # print("-------------------------------------------------------")
    # # print("ACCURACY = %.4f" % acc)
    # # print("F1-MEASURE = %.4f" % f1)
    # # print("CONFUSION MATRIX\n")
    # # print(cm)
    # # print("\n\n")
    # # print(metrics.classification_report(valdn_true_lbls, pred_labels))
    # # print("-------------------------------------------------------")
    # # print()

    # ---------------------------------------------------------------------------------------------------------------- #
    # loading a saved model to verify behavior of the model with eval_model() and predict() methods
    # model_file = os.path.abspath(os.path.join(os.getcwd(), "../results/outputs_xlBaseCased_A/"))
    # model = ClassificationModel('xlnet', model_file)

    # results = dictionary containing evaluation results (mcc, tp, tn, fp, fn)
    # model_outputs = numpy.ndarray of raw model outputs for each input
    # wrong_predictions = list of InputExample objects corresponding to each incorrect prediction by the model
    # results_e, model_outputs_e, wrong_predictions_e = model.eval_model(valdn_df, accuracy=sklearn.metrics.accuracy_score, \
    #                                                                      f1=sklearn.metrics.f1_score)
    #
    # print(model_outputs_e.shape)
    # print(model_outputs_e[0:5])
    # pd.set_option('display.max_rows', None)
    # df_e = pd.DataFrame(model_outputs_e)
    # df_e.columns = ['Column1', 'Column2']
    # writer = pd.ExcelWriter('model_rawOut_A.xlsx')
    # df_e.to_excel(writer, 'model_e_out')
    # import sys
    # sys.stdout = open('model_e_rawOut.txt', 'w')
    # print(df_obj)

    # results = python list of predictions (0 or 1) for each input
    # model_outputs = python list of raw model outputs for each input
    # results_p, model_outputs_p = model.predict(valdn_samples)
    # print(model_outputs_p.shape)
    # print(model_outputs_p[0:5])
    # pd.set_option('display.max_rows', None)
    # df_p = pd.DataFrame(model_outputs_p)
    # df_p.columns = ['Column1', 'Column2']
    # df_p.to_excel(writer, 'model_p_out')
    # writer.save()
    # import sys
    # sys.stdout = open('model_p_rawOut.txt', 'w')
    # print(df_obj)
    # ---------------------------------------------------------------------------------------------------------------- #


if __name__ == "__main__":
    main()
from GSSFiltering.model import NCLT
from GSSFiltering.filtering import Extended_Kalman_Filter
from GSSFiltering.filtering import (
    KalmanNet_Filter,
    Split_KalmanNet_Filter,
    KalmanNet_Filter_v2,
)
from GSSFiltering.trainer import Trainer
from GSSFiltering.tester import Tester
import configparser

config = configparser.ConfigParser()
config.read("./config.ini")
train_iter = config["Train"]["train_iter"]

# TRAIN = True
TRAIN = False


# S_KalmanNet
test_list = ["best"]

loss_list_Kalman = []
loss_list_Kalman_v2 = []
loss_list_Split = []
loss_ekf = []

if TRAIN:
    # KalmanNet
    trainer_kalman = Trainer(
        dnn=KalmanNet_Filter(NCLT(mode="train")),
        data_path="./.data/NCLT/train/",
        save_path="(NCLT) KalmanNet.pt",
        mode=0,
    )

    # KalmanNet (architecture 2)
    trainer_kalman_v2 = Trainer(
        dnn=KalmanNet_Filter_v2(NCLT(mode="train")),
        data_path="./.data/NCLT/train/",
        save_path="(NCLT, v2) KalmanNet.pt",
        mode=0,
    )

    # S_KalmanNet
    trainer_split = Trainer(
        dnn=Split_KalmanNet_Filter(NCLT(mode="train")),
        data_path="./.data/NCLT/train/",
        save_path="(NCLT) Split_KalmanNet.pt",
        mode=1,
    )

    for i in range(int(train_iter)):
        trainer_split.train_batch()
        trainer_split.dnn.reset(clean_history=True)
        if trainer_split.train_count % trainer_split.save_num == 0:
            trainer_split.validate(
                Tester(
                    filter=Split_KalmanNet_Filter(NCLT(mode="valid")),
                    data_path="./.data/NCLT/valid/",
                    model_path="./.model_saved/(NCLT) Split_KalmanNet_" + str(trainer_split.train_count) + ".pt",
                    is_validation=True,
                )
            )

        trainer_kalman.train_batch()
        trainer_kalman.dnn.reset(clean_history=True)
        if trainer_kalman.train_count % trainer_kalman.save_num == 0:
            trainer_kalman.validate(
                Tester(
                    filter=KalmanNet_Filter(NCLT(mode="valid")),
                    data_path="./.data/NCLT/valid/",
                    model_path="./.model_saved/(NCLT) KalmanNet_" + str(trainer_kalman.train_count) + ".pt",
                    is_validation=True,
                )
            )

        trainer_kalman_v2.train_batch()
        trainer_kalman_v2.dnn.reset(clean_history=True)
        if trainer_kalman_v2.train_count % trainer_kalman_v2.save_num == 0:
            trainer_kalman_v2.validate(
                Tester(
                    filter=KalmanNet_Filter_v2(NCLT(mode="valid")),
                    data_path="./.data/NCLT/valid/",
                    model_path="./.model_saved/(NCLT, v2) KalmanNet_" + str(trainer_kalman_v2.train_count) + ".pt",
                    is_validation=True,
                )
            )


TEST_DATA_PATH = "./.data/NCLT/test/"
tester_ekf = Tester(
    filter=Extended_Kalman_Filter(NCLT(mode="test")),
    data_path=TEST_DATA_PATH,
    model_path="EKF",
)
loss_ekf = [tester_ekf.loss.item()]
print(loss_ekf)

for elem in test_list:
    tester_skf = Tester(
        filter=Split_KalmanNet_Filter(NCLT(mode="test")),
        data_path=TEST_DATA_PATH,
        model_path="./.model_saved/(NCLT) Split_KalmanNet_" + elem + ".pt",
    )
    loss_list_Split += [tester_skf.loss.item()]

    tester_kf = Tester(
        filter=KalmanNet_Filter(NCLT(mode="test")),
        data_path=TEST_DATA_PATH,
        model_path="./.model_saved/(NCLT) KalmanNet_" + elem + ".pt",
    )
    loss_list_Kalman += [tester_kf.loss.item()]

    tester_kf2 = Tester(
        filter=KalmanNet_Filter_v2(NCLT(mode="test")),
        data_path=TEST_DATA_PATH,
        model_path="./.model_saved/(NCLT, v2) KalmanNet_" + elem + ".pt",
    )
    loss_list_Kalman_v2 += [tester_kf2.loss.item()]


print(f"EKF Loss: {loss_ekf}")
print(f"KalmanNet Loss: {loss_list_Kalman}")
print(f"KalmanNetv2 Loss: {loss_list_Kalman_v2}")
print(f"Split-KalmanNet Loss: {loss_list_Split}")

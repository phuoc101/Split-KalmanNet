from GSSFiltering.model import SyntheticNLModel
from GSSFiltering.filtering import Extended_Kalman_Filter
from GSSFiltering.filtering import KalmanNet_Filter, Split_KalmanNet_Filter, KalmanNet_Filter_v2
from GSSFiltering.trainer import Trainer
from GSSFiltering.tester import Tester
import logging
import numpy as np
import configparser
import os

logging.basicConfig(format="[%(levelname)s]: %(message)s", level=logging.INFO)

if not os.path.exists("./.results"):
    os.mkdir("./.results")

config = configparser.ConfigParser()
config.read("./config.ini")

TRAIN = True
# TRAIN = False

train_iter = int(config["Train"]["train_iter"])

# S_KalmanNet
test_list = ["500"]

loss_list_Kalman = []
loss_list_Kalman_v2 = []
loss_list_Split = []
loss_ekf = []

valid_loss_Kalman = []
valid_loss_Kalman_v2 = []
valid_loss_Split = []

if TRAIN:
    # Data generation
    # Train model
    train_syntheticnl_model = SyntheticNLModel(mode="train")
    train_syntheticnl_model.init_params()
    train_syntheticnl_model.generate_data()

    valid_syntheticnl_model = SyntheticNLModel(mode="train")
    valid_syntheticnl_model.init_params()
    valid_syntheticnl_model.generate_data()

    # KalmanNet
    trainer_kalman = Trainer(
        dnn=KalmanNet_Filter(train_syntheticnl_model),
        data_path="./.data/syntheticNL/train/",
        save_path="(syntheticNL) KalmanNet.pt",
        mode=0,
    )

    # KalmanNet (architecture 2)
    trainer_kalman_v2 = Trainer(
        dnn=KalmanNet_Filter_v2(train_syntheticnl_model),
        data_path="./.data/syntheticNL/train/",
        save_path="(syntheticNL, v2) KalmanNet.pt",
        mode=0,
    )

    # S_KalmanNet
    trainer_split = Trainer(
        dnn=Split_KalmanNet_Filter(train_syntheticnl_model),
        data_path="./.data/syntheticNL/train/",
        save_path="(syntheticNL) Split_KalmanNet.pt",
        mode=1,
    )

    for i in range(int(train_iter)):

        trainer_split.train_batch()
        trainer_split.dnn.reset(clean_history=True)
        if trainer_split.train_count % trainer_split.save_num == 0:
            trainer_split.validate(
                Tester(
                    filter=Split_KalmanNet_Filter(valid_syntheticnl_model),
                    data_path="./.data/syntheticNL/valid/",
                    model_path="./.model_saved/(syntheticNL) Split_KalmanNet_" + str(trainer_split.train_count) + ".pt",
                    is_validation=True,
                )
            )
            valid_loss_Split += [trainer_split.valid_loss]

        trainer_kalman.train_batch()
        trainer_kalman.dnn.reset(clean_history=True)
        if trainer_kalman.train_count % trainer_kalman.save_num == 0:
            trainer_kalman.validate(
                Tester(
                    filter=KalmanNet_Filter(valid_syntheticnl_model),
                    data_path="./.data/syntheticNL/valid/",
                    model_path="./.model_saved/(syntheticNL) KalmanNet_" + str(trainer_kalman.train_count) + ".pt",
                    is_validation=True,
                )
            )
            valid_loss_Kalman += [trainer_kalman.valid_loss]

        trainer_kalman_v2.train_batch()
        trainer_kalman_v2.dnn.reset(clean_history=True)
        if trainer_kalman_v2.train_count % trainer_kalman_v2.save_num == 0:
            trainer_kalman_v2.validate(
                Tester(
                    filter=KalmanNet_Filter_v2(valid_syntheticnl_model),
                    data_path="./.data/syntheticNL/valid/",
                    model_path="./.model_saved/(syntheticNL, v2) KalmanNet_"
                    + str(trainer_kalman_v2.train_count)
                    + ".pt",
                    is_validation=True,
                )
            )
            valid_loss_Kalman_v2 += [trainer_kalman_v2.valid_loss]

    validator_ekf = Tester(
        filter=Extended_Kalman_Filter(valid_syntheticnl_model),
        data_path="./.data/syntheticNL/valid/",
        model_path="EKF",
    )
    loss_ekf = [validator_ekf.loss.item()]

    np.save("./.results/valid_loss_ekf.npy", np.array(loss_ekf))
    np.save("./.results/valid_loss_kalman.npy", np.array(valid_loss_Kalman))
    np.save("./.results/valid_loss_kalman_v2.npy", np.array(valid_loss_Kalman_v2))
    np.save("./.results/valid_loss_split.npy", np.array(valid_loss_Split))

# Testing
test_syntheticnl_model = SyntheticNLModel(mode="test")
test_syntheticnl_model.init_params()
test_syntheticnl_model.generate_data()

tester_ekf = Tester(
    filter=Extended_Kalman_Filter(test_syntheticnl_model),
    data_path="./.data/syntheticNL/test/",
    model_path="EKF",
)
loss_ekf = [tester_ekf.loss.item()]
print(loss_ekf)

for elem in test_list:

    tester_kf = Tester(
        filter=KalmanNet_Filter(test_syntheticnl_model),
        data_path="./.data/syntheticNL/test/",
        model_path="./.model_saved/(syntheticNL) KalmanNet_" + elem + ".pt",
    )
    loss_list_Kalman += [tester_kf.loss.item()]

    tester_kf2 = Tester(
        filter=KalmanNet_Filter_v2(test_syntheticnl_model),
        data_path="./.data/syntheticNL/test/",
        model_path="./.model_saved/(syntheticNL, v2) KalmanNet_" + elem + ".pt",
    )
    loss_list_Kalman_v2 += [tester_kf2.loss.item()]

    tester_skf = Tester(
        filter=Split_KalmanNet_Filter(test_syntheticnl_model),
        data_path="./.data/syntheticNL/test/",
        model_path="./.model_saved/(syntheticNL) Split_KalmanNet_" + elem + ".pt",
    )
    loss_list_Split += [tester_skf.loss.item()]

logging.info(f"EKF Loss: {loss_ekf}")
logging.info(f"KalmanNetv1 Loss: {loss_list_Kalman}")
logging.info(f"KalmanNetv2 Loss: {loss_list_Kalman_v2}")
logging.info(f"Split-KalmanNet Loss: {loss_list_Split}")

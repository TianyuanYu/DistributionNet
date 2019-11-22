python train_ReID_classifier_from_resnet50.py --set bounding_box_train_0.1_0 --standard True

python train_ReID_classifier_con.py --max_number_of_steps 20200 --target market --dataset_name Market --set bounding_box_train_0.1_0  --checkpoint_path2 ./result/resnet_v1_50_targetmarket_noise_0.1_0_standard

python train_ReID_classifier.py --entropy_loss True --max_number_of_steps 20200 --target market --dataset_name Market --set bounding_box_train_0.1_0 --checkpoint_path2 ./result/resnet_v1_50_targetmarket_noise_0.1_0_standard

python eval_market_multi.py

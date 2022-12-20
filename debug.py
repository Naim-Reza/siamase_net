import torch.utils.data
from tqdm import tqdm

from DataLoader import SiameseDataset

if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_root = '/media/naim/4A62E7E862E7D6AB/Users/chosun/Datasets/test_set/'
    weights_root = './weights'
    LOG_DIR = './logs'

    input_size = 112
    batch_size = 4
    input_feature_size = 2048
    embedding_size = 512
    num_workers = 4
    pin_memory = True

    # ==== DataLoader === #
    test_dataset = SiameseDataset(data_root, image_size=input_size, test=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size, num_workers=num_workers,
                                                  pin_memory=pin_memory)

    # === Perform Testing === #
    positive_distances = list()
    negative_distances = list()
    prev_class_name = 0
    positive_dict = {}
    negative_dict = {}
    for a, p, n, class_names in tqdm(iter(test_dataloader), total=len(test_dataloader)):
        anchore, positive, negative = a.to(device), p.to(device), n.to(device)
        # anchore_features, positive_features, negative_features = backbone(anchore), backbone(positive), backbone(
        #     negative)
        #
        # positive_distance, negative_distance = head(anchore_features, positive_features, negative_features)
        # pds = positive_distance.detach().cpu().numpy()
        # nds = negative_distance.detach().cpu().numpy()
        # # print(class_names)

        for i, class_name in enumerate(class_names):
            # if int(class_name) == 98 or int(class_name) == 99:
            #     print(f'classname found: {class_name}')
            if not int(class_name) == int(prev_class_name):
                # print(f'Previous class: {prev_class_name}, Current class: {class_name}')
                positive_dict[prev_class_name] = positive_distances
                negative_dict[prev_class_name] = negative_distances
                prev_class_name = int(class_name)
                positive_distances = []
                negative_distances = []

            # positive_distances.append(pds[i])
            # negative_distances.append(nds[i])
            # positive_dict[int(class_name)] = positive_distances
            # negative_dict[int(class_name)] = negative_distances

        # print(positive_distances)

    # log_file = os.path.join(LOG_DIR, f'test_{args.logFileName}_logs.npy')
    # print(f'saving test result logs in {log_file}')
    # print(positive_dict)
    #
    # np.save(log_file, np.array([positive_dict, negative_dict]))
    # print(f'Test log saved in {log_file}')

    # print(positive_dict)
    # print(negative_dict)

import torch
from dataset import get_data_transforms
from resnet_TTA import  wide_resnet50_2
from de_resnet import  de_wide_resnet50_2
from dataset import MVTecDataset, MVTecDatasetOOD
from test import  evaluation_ATTA



def test_mvtec(_class_):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Class: ', _class_)
    data_transform, gt_transform = get_data_transforms(128, 128)

    #load data
    test_path_id = './visa/' + _class_ #update here
    test_path_brightness = './visa_brightness/' + _class_ #update here
    test_path_constrast = './visa_contrast/' + _class_ #update here
    test_path_defocus_blur = './visa_defocus_blur/' + _class_ #update here
    test_path_gaussian_noise = './visa_gaussian_noise/' + _class_ #update here
    ckp_path = './checkpoints/' + 'visa_DINL_' + str(_class_) + '_19.pth'
    test_data_id = VisADataset(root=test_path_id, transform=data_transform, gt_transform=gt_transform,
                             phase="test")
    test_data_brightness = VisADatasetOOD(root=test_path_brightness, transform=data_transform, gt_transform=gt_transform,
                             phase="test", _class_=_class_)
    test_data_constrast = VisADatasetOOD(root=test_path_constrast, transform=data_transform, gt_transform=gt_transform,
                             phase="test", _class_=_class_)
    test_data_defocus_blur = VisADatasetOOD(root=test_path_defocus_blur, transform=data_transform, gt_transform=gt_transform,
                             phase="test", _class_=_class_)
    test_data_gaussian_noise = VisADatasetOOD(root=test_path_gaussian_noise, transform=data_transform, gt_transform=gt_transform,
                             phase="test", _class_=_class_)

    test_dataloader_id = torch.utils.data.DataLoader(test_data_id, batch_size=1, shuffle=False)
    test_dataloader_brightness = torch.utils.data.DataLoader(test_data_brightness, batch_size=1, shuffle=False)
    test_dataloader_constrast = torch.utils.data.DataLoader(test_data_constrast, batch_size=1, shuffle=False)
    test_dataloader_defocus_blur = torch.utils.data.DataLoader(test_data_defocus_blur, batch_size=1, shuffle=False)
    test_dataloader_gaussian_noise = torch.utils.data.DataLoader(test_data_gaussian_noise, batch_size=1, shuffle=False)

    #load model
    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)

    #load checkpoint
    ckp = torch.load(ckp_path)
    for k, v in list(ckp['bn'].items()):
        if 'memory' in k:
            ckp['bn'].pop(k)
    decoder.load_state_dict(ckp['decoder'])
    bn.load_state_dict(ckp['bn'])

    lamda = 0.5

    list_results = []
    auroc_sp = evaluation_ATTA(encoder, bn, decoder, test_dataloader_id, device,
                                               type_of_test='EFDM_test',
                                               img_size=128, lamda=lamda, dataset_name='visa', _class_=_class_)
    list_results.append(round(auroc_sp, 4))
    print('Auroc of ID data{:.4f}'.format(auroc_sp))

    auroc_sp = evaluation_ATTA(encoder, bn, decoder, test_dataloader_brightness, device,
                                               type_of_test='EFDM_test',
                                               img_size=128, lamda=lamda, dataset_name='visa', _class_=_class_)
    list_results.append(round(auroc_sp, 4))
    print('Auroc of brightness data{:.4f}'.format(auroc_sp))

    auroc_sp = evaluation_ATTA(encoder, bn, decoder, test_dataloader_constrast, device,
                                               type_of_test='EFDM_test',
                                               img_size=128, lamda=lamda, dataset_name='visa', _class_=_class_)
    list_results.append(round(auroc_sp, 4))
    print('Auroc of contrast data{:.4f}'.format(auroc_sp))

    auroc_sp = evaluation_ATTA(encoder, bn, decoder, test_dataloader_defocus_blur, device,
                                               type_of_test='EFDM_test',
                                               img_size=128, lamda=lamda, dataset_name='visa', _class_=_class_)
    list_results.append(round(auroc_sp, 4))
    print('Auroc of defocus blur data{:.4f}'.format(auroc_sp))

    auroc_sp = evaluation_ATTA(encoder, bn, decoder, test_dataloader_gaussian_noise, device,
                                               type_of_test='EFDM_test',
                                               img_size=128, lamda=lamda, dataset_name='visa', _class_=_class_)
    list_results.append(round(auroc_sp, 4))
    print('Auroc of Gaussian noise data{:.4f}'.format(auroc_sp))

    print(list_results)

    return

item_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
             'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood']
for i in item_list:
    test_mvtec(i)
    print('===============================================')
    print('')
    print('')


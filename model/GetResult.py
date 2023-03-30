from __future__ import print_function
import torch
import torch.nn
from model.IviuNet import IVIUNet
from model.Dataloader import get_loaded_data
from model.ParaSet import ParaSet


def get_result(para: ParaSet):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loaded_data, lib_data = get_loaded_data(para.batch_size, para.data_dir, para.lib_path,
                                            para.lib_symbol)
    lib_data = lib_data.float().to(device)

    model = IVIUNet(lib_data, para.length, para.width, para.admm_layers).to(device)

    checkpoint = torch.load(para.ckpt, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    abuList, curveList, sreList = [], [], []
    with torch.no_grad():
        for idx, (mea_data, target) in enumerate(loaded_data):
            mea_data = mea_data.float().cuda()
            est_abu, rec_curve = model(mea_data)  # est_abu为丰度, rec_curve为重构结果
            if not para.train_from and para.date_name != 'Cuprite':
                abuList.append(est_abu)
                curveList.append(rec_curve)
                sreList.append((10 * torch.log10(
                    torch.pow(torch.norm(target, 2), 2) / torch.pow(
                        torch.norm(target - est_abu.cpu(), 2), 2))).numpy())
            else:
                abuList = est_abu
                curveList = rec_curve
                
    return abuList, curveList, sreList
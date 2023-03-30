from statistics import mean
import wandb
import torch
import torch.optim as optim
from model.IviuNet import IVIUNet
from model.Dataloader import get_loaded_data
from model.ModelLoss import ModelLoss
from model.ParaSet import ParaSet


def train(para: ParaSet):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loaded_data, lib_data = get_loaded_data(para.batch_size, para.data_dir,
                                            para.lib_path, para.lib_symbol)
    min_loss = float('inf')
    lib_data = lib_data.to(device)
    model = IVIUNet(lib_data, para.length, para.width, para.admm_layers).to(device)
    optimizer = optim.Adam(
        model.parameters(),
        para.learning_rate,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                     para.epoch,
                                                     eta_min=0)

    if para.train_from:
        checkpoint = torch.load(para.train_from)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    # 由于参数分析的实验是后加的，因此默认损失函数的权重都是 1
    model_loss = ModelLoss(lib_data, alpha1=1, alpha2=1, alpha3=1)
    for epoch in range(para.epoch):
        for idx, (mea_data, targetAbu) in enumerate(loaded_data):
            mea_data = mea_data.float().cuda()
            est_abu, rec_curve = model(mea_data)
            loss_dict = model_loss(rec_curve, mea_data)
            loss = sum(loss_dict.values())
            if torch.isnan(loss):
                print(loss_dict)
                raise Exception()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        if (epoch + 1) % 5 == 0:
            print('Epoch {:.0f} / {:.0f}, Loss:{:.8f}, LR:{:.10f}'.format(
                epoch + 1, para.epoch, loss.item(),
                scheduler.get_last_lr()[0]),
                  end=' ')
            print(" ".join([
                f"{loss_name}: {loss_val:.8f}"
                for (loss_name, loss_val) in loss_dict.items()
            ]))
            sre = (10 * torch.log10(
                    torch.pow(torch.norm(targetAbu, 2), 2) / torch.pow(
                        torch.norm(targetAbu - est_abu.cpu(), 2), 2)))
            print(sre)
            
            # wandb 记录运行数据
            if not para.train_from and para.length == 75:
                wandb.log({
                    **loss_dict, "lr": scheduler.get_last_lr()[0],
                    "loss": loss, 'sre': sre
                })

        if loss < min_loss:
            min_loss = loss
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                },
                f"{para.save_dir}/{para.network}_{para.date_name}_min_loss.pt")

    if not para.train_from:
        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, f"{para.save_dir}/{para.network}_{para.date_name}_final.pt")

import torch
from torch.cuda.amp import GradScaler, autocast

from metrics import PSNR, SSIM


class Generic_train_test():
    def __init__(self, model, opts, dataloader, val_dataloader):
        self.model = model
        self.opts = opts
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.scaler = GradScaler()  # 初始化GradScaler
        self.fake_batchsize = opts.fake_batchsize  # 假设opts中有一个fake_batchsize属性

    def decode_input(self, data):
        raise NotImplementedError()

    def validate(self):
        """在验证集上评估模型性能。"""
        self.model.net_G.eval()  # 设置为评估模式
        total_steps = 0
        val_loss = 0.0
        val_ssim = 0.0
        val_psnr = 0.0
        with torch.no_grad():  # 关闭梯度计算
            for _, data in enumerate(self.val_dataloader):
                total_steps += 1
                _input = self.decode_input(data)
                with autocast():
                    pred_Cloudfree_data = self.model.forward()
                    loss_G = self.model.loss_fn(pred_Cloudfree_data, self.model.cloudfree_data)
                    psnr = PSNR(pred_Cloudfree_data, self.model.cloudfree_data)
                    ssim = SSIM(pred_Cloudfree_data, self.model.cloudfree_data)
                    if total_steps % self.opts.log_freq == 0:
                        print('steps', total_steps, 'val_loss', loss_G, 'psnr', psnr, 'ssim', ssim)
                val_loss += loss_G.item()
                val_ssim += ssim.item()
                val_psnr += psnr
        val_loss /= len(self.val_dataloader)
        val_psnr /= len(self.val_dataloader)
        val_ssim /= len(self.val_dataloader)
        self.model.net_G.train()  # 将模型设置回训练模式
        return val_loss, val_psnr, val_ssim

    def train(self):
        total_steps = 0
        print('#training images ', len(self.dataloader) * self.opts.batch_sz)

        log_loss = 0
        steps_per_update = max(self.fake_batchsize // self.opts.batch_sz, 1)  # 计算梯度累积步数
        accumulation_steps = 0  # 初始化累积步数
        for epoch in range(self.opts.max_epochs):
            print(f'epoch: {epoch}')
            self.model.optimizer_G.zero_grad()  # 确保在epoch开始时梯度被清零

            for _, data in enumerate(self.dataloader):
                total_steps += 1
                _input = self.decode_input(data)

                with autocast():
                    # 替换原有的optimize_parameters调用
                    self.model.set_input(_input)  # 设置输入
                    pred_Cloudfree_data = self.model.forward()  # 前向传播
                    self.model.update_info(pred_Cloudfree_data)
                    loss_G = self.model.loss_fn(pred_Cloudfree_data, self.model.cloudfree_data)  # 计算损失

                # 缩放损失并执行反向传播，但在累积足够梯度前不进行参数更新
                self.scaler.scale(loss_G).backward()

                if (total_steps % steps_per_update == 0) or (_ == len(self.dataloader) - 1):
                    # 更新模型参数
                    self.scaler.step(self.model.optimizer_G)
                    self.scaler.update()
                    self.model.optimizer_G.zero_grad()  # 清空梯度
                    accumulation_steps = 0  # 重置累积步数
                else:
                    accumulation_steps += 1

                log_loss += loss_G.item()

                # =========== visualize results ============#
                if total_steps % self.opts.log_freq == 0:
                    info = self.model.get_current_scalars()
                    print('epoch', epoch, 'steps', total_steps, 'loss', log_loss / self.opts.log_freq, info)
                    log_loss = 0

            if epoch % self.opts.save_freq == 0:
                print('validating~~')
                val_loss, val_psnr, val_ssim = self.validate()  # 进行验证
                print(f'Validation Loss after epoch {epoch}: {val_loss},  PSNR:{val_psnr},SSSIM:{val_ssim}')

                self.model.save_checkpoint(epoch)

            if epoch > self.opts.lr_start_epoch_decay - self.opts.lr_step:
                self.model.update_lr()

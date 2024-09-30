from imports import *
from CogDataset3d import *
import numpy as np
import h5py
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

#seed_everything(11)

from torchmetrics import R2Score as _r2score
#from ignite.contrib.metrics.regression import R2Score


from collections import OrderedDict
#seed_everything(11)

device = 'cuda'

PARAMS = {
    'min_epochs': 1,
    # Change according to your gpu settings.
    'max_epochs': 45,
    'learning_rate': 1e-5,
    'batch_size': 5,
    'weight_decay' : 1e-5,
    # If required for RFA.
    'extract_features' : False
}


class LitURNet3d(pl.LightningModule):
    def __init__(self,
                 in_channels=1,
                 num_classes=4, 
                 batch_size=PARAMS['batch_size'],
                 lr=PARAMS['learning_rate'],
                 weight_decay=PARAMS['weight_decay']):
        super(LitURNet3d, self).__init__()
        
        self.automatic_optimization = False
        self.df = pd.read_csv('cleaned_df_5_31.csv')
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.final_w1_epoch = 0.5
        self.final_w2_epoch = 0.5

        self.cb1 = contraction_block3d(in_ch=1, out_ch=16)
        self.cb2 = contraction_block3d(in_ch=16, out_ch=32)
        self.cb3 = contraction_block3d(in_ch=32, out_ch=64)
        self.cb4 = contraction_block3d(in_ch=64, out_ch=128)
        self.convb = conv_block3d(in_ch=128, out_ch=256)
        self.regb = regression_block3d(in_ch=256, out_ch=128)  
        self.dropout = nn.Dropout(0.15)
        # can be altered
        self.linear = nn.Linear(128*8*8, 1) 
        self.exb1 = expansion_block3d(in_ch=256, out_ch=128)
        self.exb2 = expansion_block3d(in_ch=128, out_ch=64)
        self.exb3 = expansion_block3d(in_ch=64, out_ch=32)
        self.exb4 = expansion_block3d(in_ch=32, out_ch=16)

        self.final_conv = nn.Conv3d(in_channels=16, out_channels=4, kernel_size=3, padding=1, stride=1)
 
        
    def forward(self, X, extract_features=True):
        extracted = {}
        cb1_out, out1 = self.cb1(X)
        if extract_features: extracted["out_1"]=out1
        cb2_out, out2 = self.cb2(out1)
        if extract_features: extracted["out_2"]=out2
        cb3_out, out3 = self.cb3(out2)
        if extract_features: extracted["out_3"]=out3
        cb4_out, out4 = self.cb4(out3)
        if extract_features: extracted["out_4"]=out4
        out5 = self.convb(out4)
        if extract_features: extracted["out_5"]=out5
        regb_out = self.regb(out5)
        drp_out = self.dropout(regb_out)
        # can I leave the if here?
        if extract_features: extracted["drp_out"] = drp_out
        reg_x =  drp_out.view(-1,8192) #CHANGED
        if extract_features: extracted["reg_in"]=reg_x
        reg_out = self.linear(reg_x)    #CHANGED
        if extract_features: extracted["reg_out"]=reg_out
        out6 = self.exb1(cb4_out, out5)
        if extract_features: extracted["out_6"]=out6
        out7 = self.exb2(cb3_out, out6)
        if extract_features: extracted["out_7"]=out7
        out8 = self.exb3(cb2_out, out7)
        if extract_features: extracted["out_8"]=out8
        out9 = self.exb4(cb1_out, out8)
        if extract_features: extracted["out_9"]=out9
        seg_out = self.final_conv(out9)
        if extract_features: extracted["seg_out"]=seg_out
        if extract_features:

            return reg_out, seg_out, extracted
        return reg_out, seg_out, extracted

        
        
    def training_step(self, train_batch, batch_idx):
        X, y_img, y_adas, filename = train_batch
        dim1,dim2,dim3,dim4 = y_img.size()
        
        total_pixels = dim1*dim2*dim3*dim4
        X = X.view(dim1,1,dim2,dim3,dim4)
        y_img = y_img.squeeze(1).long()
        y_adas = y_adas.unsqueeze(1).float()
        adas_out, seg_out, extracted = self(X)   

        adas_out_combined, weight1, weight2 = self.get_combined_preds(self.df, 
                                                    adas_out, y_adas, filename, False)
        adas_out_combined = adas_out_combined.type_as(X)

        linear_pred = adas_out_combined.view(-1).cuda()     

        # segmentation metrics
        seg_loss = F.cross_entropy(seg_out, y_img)
        _, preds = torch.max(seg_out.data, 1)
        y_seg_acc = preds.eq(y_img).sum().item()/total_pixels

        # regression metrics
        adas_loss = F.smooth_l1_loss(linear_pred, y_adas.squeeze(1))
        adas_mse_loss = F.mse_loss(linear_pred, y_adas.squeeze(1))
        r2_score = _r2score()
        r2_score.to(device)
        y_adas_squeeze = y_adas.squeeze(1)
        adas_r2 = r2_score(linear_pred,y_adas_squeeze)
        
        # custom loss function

        #adas_r2 = _r2score(linear_pred, y_adas.squeeze(1))
    
        opt = self.optimizers()
        opt.zero_grad()
        loss = seg_loss + adas_loss  
        self.manual_backward(loss)
        opt.step()
        
        self.log('train_total_loss', loss, on_step=False, on_epoch=True,
                 sync_dist=True, logger=True)
#         self.log('train_extracted', extracted, on_step=True, on_epoch=True,
#                  sync_dist=True, logger=True) # fm
        self.log('train_score_loss', adas_loss, on_step=False, on_epoch=True,
                 sync_dist=True, logger=True)
        self.log('train_r2', adas_r2, on_step=False, on_epoch=True,
                 sync_dist=True, logger=True)
        self.log('train_score_mse_loss', adas_mse_loss, on_step=False, on_epoch=True,
                 sync_dist=True, logger=True)
#         with open('fm.json', 'w') as fm:
#             fm.write(json.dumps(extracted))
#         np.savez("extracted.npz", **extracted)
        hf = h5py.File(f'/media/rajlab/sachin_data_1/userdata/daren/feature_map/seg_out{batch_idx}.hdf5','w')
#         for block in ["out_1", "out_2", "out_3", "out_4", "out_5", "drp_out", "reg_in", "reg_out", "out_6", "out_7", "out_8", "out_9"]:
        for block in ["seg_out"]:
            hf[block] = extracted[block].cpu().detach().numpy()
        hf.close()

        return {'loss': loss, 
                'adas_loss': adas_loss, 
                'adas_mse_loss': adas_mse_loss, 
                'adas_r2': adas_r2, 
#                 'extracted':extracted, # fm
                'w1': weight1, 
                'w2': weight2} 

    
    
    def training_epoch_end(self, training_step_outputs):
        train_r2s = []
        train_losses = []
        train_score_losses = []
        train_score_mse_losses = []
        # add FM
        train_features = []

        w1s = []
        w2s = []

        for train_out in training_step_outputs:
            train_losses.append(train_out['loss'])
            train_r2s.append(train_out['adas_r2'])
            train_score_losses.append(train_out['adas_loss'])
            train_score_mse_losses.append(train_out['adas_mse_loss'])
            train_features.append(train_out['extracted']) # fm
            if train_out['adas_r2'] > 0.10:
                w1s.append(train_out['w1'])
                w2s.append(train_out['w2'])


        train_r2_mean = (torch.stack(train_r2s).mean())
        train_loss_mean = (torch.stack(train_losses).mean())
        train_score_mse_loss_mean = (torch.stack(train_score_mse_losses).mean())
        train_loss_score_mean = (torch.stack(train_score_losses).mean())

        
        if len(w1s) > 0:        
            self.final_w1_epoch = sum(w1s)/len(w1s)
            self.final_w2_epoch = sum(w2s)/len(w2s)
        else:
            self.final_w1_epoch = 0.5
            self.final_w2_epoch = 0.5
        
        # was in training_steps

        
        self.log('train_loss_epoch', train_loss_mean,  logger=True)
        self.log('train_score_mse_loss_epoch', train_score_mse_loss_mean,  logger=True)
        self.log('train_score_loss_epoch', train_loss_score_mean,  logger=True)
        self.log('train_r2_epoch', train_r2_mean,  logger=True)
#         with open('fm.json', 'w') as fm:
#             fm.write(json.dumps(extracted))
    

    
    def validation_step(self, val_batch, batch_idx):
        X, y_img, y_adas, filename = val_batch

        dim1,dim2,dim3,dim4 = y_img.size()
        total_pixels        = dim1*dim2*dim3*dim4

        X = X.view(dim1,1,dim2,dim3,dim4)
        y_img = y_img.squeeze(1).long()
        y_adas = y_adas.unsqueeze(1).float()
        adas_out, seg_out, extracted = self(X) # fm
#         adas_out, seg_out = self(X)
        adas_out_combined, _, _ = self.get_combined_preds(self.df, adas_out, y_adas, 
                                            filename, True, self.final_w1_epoch, self.final_w2_epoch)
        
        adas_out_combined = adas_out_combined.type_as(X)
        linear_pred = adas_out_combined.view(-1).cuda()  

        seg_loss = F.cross_entropy(seg_out, y_img)
        _, preds = torch.max(seg_out.data, 1)
        y_seg_acc = preds.eq(y_img).sum().item()/total_pixels

        vadas_loss = F.smooth_l1_loss(linear_pred, y_adas.squeeze(1))
        vadas_mse_loss = F.mse_loss(linear_pred, y_adas.squeeze(1))
        r2_score = _r2score()
        #device = 'cuda'
        r2_score.to(device)
        #print(linear_pred.get_device())
        y_adas_squeeze = y_adas.squeeze(1)
        #print(y_adas_squeeze.get_device())
        vadas_r2 = r2_score(linear_pred,y_adas_squeeze)

        # combined loss
        vloss = seg_loss + vadas_loss  
        
        
        self.log('val_total_loss', vloss, on_step=False, on_epoch=True,
                 sync_dist=True, logger=True)
        self.log('val_score_loss', vadas_loss, on_step=False, on_epoch=True,
                 sync_dist=True, logger=True)
        self.log('val_r2', vadas_r2, on_step=False, on_epoch=True,
                 sync_dist=True, logger=True)
        self.log('val_score_mse_loss', vadas_mse_loss, on_step=False, on_epoch=True,
                 sync_dist=True, logger=True)
    
        return {'vloss': vloss, 
                'vadas_loss': vadas_loss, 
                'vadas_mse_loss': vadas_mse_loss, 
                'vadas_r2': vadas_r2} 
                
                
        
    def validation_epoch_end(self, validation_step_outputs):
        val_r2s = []
        val_losses = []
        val_score_losses = []
        val_score_mse_losses = []
        
        for val_out in validation_step_outputs:
            val_r2s.append(val_out['vadas_r2'])
            val_losses.append(val_out['vloss'])
            val_score_losses.append(val_out['vadas_loss'])
            val_score_mse_losses.append(val_out['vadas_mse_loss'])
            
        val_loss_mean = (torch.stack(val_losses).mean())
        val_score_loss_mean = (torch.stack(val_score_losses).mean())
        val_score_mse_loss_mean = (torch.stack(val_score_mse_losses).mean())
        val_r2_mean = (torch.stack(val_r2s).mean())
        
        # print('mean train r2')
        # print(val_r2_mean)
        
        
        self.log('val_loss_epoch', val_loss_mean,  logger=True)
        self.log('val_score_loss_epoch', val_score_loss_mean,  logger=True)
        self.log('val_score_mse_loss_epoch', val_score_mse_loss_mean,  logger=True)
        self.log('val_r2_epoch', val_r2_mean,  logger=True)


        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
    
    def train_dataloader(self):
#         X_tr, X_v = get_file_splits()
#         X_train, X_val, y_adas_train, y_adas_val, y_mmse_train, y_mmse_val = split_train_val(X_tr, X_v, self.df)
        ds_train, ds_val, dl_train, dl_val = get_ds_dl('all', batch_size=self.batch_size, num_workers=16)
        return dl_train
    
    def val_dataloader(self):
        
#         X_tr, X_v = get_file_splits()
#         X_train, X_val, y_adas_train, y_adas_val, y_mmse_train, y_mmse_val = split_train_val(X_tr, X_v, self.df)
        ds_train, ds_val, dl_train, dl_val = get_ds_dl('all', batch_size=self.batch_size, num_workers=16)
        
        return dl_val
    
    
    def combined_pred(self, actual, pred1, pred2, val=False):
        
        r2_score_1 = r2_score(actual, pred1)
        r2_score_2 = r2_score(actual, pred2)

        mse_1 = mean_squared_error(actual, pred1)
        mse_2 = mean_squared_error(actual, pred2)
        
        percent_diff = (r2_score_1 - r2_score_2)/(r2_score_1)
        #percent_diff = (mse_1 - mse_2)/(mse_1)


        weight_1 = 1 / (2 - percent_diff)
        weight_2 = 1 - weight_1
        return pred1*weight_1 + pred2*weight_2, weight_1, weight_2


    def get_combined_preds(self, df, reg_out, y_scores, filenames, val=False, w1=0, w2=0):
        unet_preds = np.array([unet_pred[0].item() for unet_pred in reg_out])
        X = df[df['filenames'].isin(filenames)]
        files_in_df = X['filenames'].tolist()
        
        # Reads the customized loss gb trees.
        hgradboost_preds = []
        hgb_pipe = load('/home/madar/unet2021/hgb_model.joblib') 

        for idx, f in enumerate(filenames):
            if f in files_in_df:
                X = df[df['filenames'] == f]
                X_drop = X.drop(columns=['filenames'])
                x = X_drop.loc[:, ((X_drop.columns != 'ADAS11') & (X_drop.columns != 'MMSE'))]
                hgradboost_preds.append(hgb_pipe.predict(x)[0])
            else:
                hgradboost_preds.append(unet_preds[idx])

        list_all_preds = [hgradboost_preds]
        list_preds_weighted = [np.array(pred)*(1) for pred in list_all_preds]
        tab_preds = np.array(sum(list_preds_weighted))

        actual = np.array(y_scores.tolist())
        if val == False:
            tab_unet_preds, w1, w2 = self.combined_pred(actual, tab_preds, unet_preds, val)

        if val == True:
            tab_unet_preds = tab_preds*w1 + unet_preds*w2
            r2_score_1 = r2_score(actual, tab_preds)
            r2_score_2 = r2_score(actual, unet_preds)
            print(f'Tab R2 : {r2_score_1}')
            print(f'UNet R2: {r2_score_2}')
            
            

        return torch.Tensor(tab_unet_preds), w1, w2

    
    
class conv_block3d(pl.LightningModule):
    def __init__(self, in_ch, out_ch, same=True):
        super(conv_block3d, self).__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1, stride=1)  
        self.bn1 = nn.BatchNorm3d(out_ch, track_running_stats = True)                        
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, stride=1) 
        self.bn2 = nn.BatchNorm3d(out_ch, track_running_stats = True)                          
        
    def forward(self, X):
        X1 = F.relu(self.bn1(self.conv1(X)))
        X2 = F.relu(self.bn2(self.conv2(X1)))
        return X2

class regression_block3d(pl.LightningModule):
    def __init__(self, in_ch, out_ch):
        super(regression_block3d, self).__init__()
        self.convb = conv_block3d(in_ch, out_ch)
        self.poolreg = nn.MaxPool3d(kernel_size=2, stride=2)  
        self.bnreg = nn.BatchNorm3d(out_ch, track_running_stats = True)                        
    
    def forward(self, X):
        poolreg_out = self.poolreg(self.convb(X))
        bnreg_out = self.bnreg(poolreg_out)
        return bnreg_out 
    
class contraction_block3d(pl.LightningModule):
    def __init__(self, in_ch, out_ch):
        super(contraction_block3d, self).__init__()                                           
        self.convb = conv_block3d(in_ch, out_ch)              
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)            
        
    def forward(self, X):
        conv_out = self.convb.forward(X)
        maxpool_out = self.pool(conv_out)
        return conv_out,maxpool_out
    
class expansion_block3d(pl.LightningModule):
    def __init__(self, in_ch, out_ch):
        super(expansion_block3d, self).__init__()
        self.transpose_conv = nn.ConvTranspose3d(in_ch, out_ch, 2, 2) 
        self.convb = conv_block3d(in_ch, out_ch)
          
    def forward(self, contraction_out, X):
        transpose_conv_out = self.transpose_conv(X)
        concat_input = torch.cat((contraction_out,transpose_conv_out), dim=1)
        conv_out = self.convb(concat_input)
        return conv_out
    




# Neptune Logger
#neptune_logger = NeptuneLogger( 
    #tags=['only_unet', 'huber_loss', 'large_unet', 'linear'],
    #api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0"+\
    #"cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3ZTAwNTc5NC0yYjI4LTQzNjgtOWRiZS1lYTAxZTU5NDhjNjQifQ==",
    #params=PARAMS,
    #project_name = 'cpabalan/cog-regression',
    #close_after_fit=True
#)


# Initalizing callbacks

if __name__ == '__main__':
    checkpoint_callback = ModelCheckpoint(
        monitor='val_score_loss_epoch',
        dirpath='/home/madar/unet2021/models/lightning_models/with_tabular/',
        filename = '{val_r2_epoch:.4f}_{val_score_mse_loss_epoch:.2f}_{epoch:02d}',
        save_top_k=3,
        mode = 'min'
    )


    trainer = Trainer(
     #   fast_dev_run = True,
        gpus=[0], 
#         auto_select_gpus=True,
        #auto_lr_find=True,
#         strategy='ddp',
        precision=16,
        #deterministic=True,
        plugins=DDPPlugin(find_unused_parameters=True),
        callbacks=[checkpoint_callback],
        min_epochs= PARAMS['min_epochs'],
        max_epochs = PARAMS['max_epochs'],
#         extract_features = PARAMS['extract_features']
        #logger=neptune_logger
    )


    #model = LitURNet3d(1,4)
    PATH_TO_MODEL = '/home/madar/unet2021/models/lightning_models/with_tabular/val_r2_epoch=0.5943_val_score_mse_loss_epoch=31.61_epoch=00.ckpt'#model = model.load_from_checkpoint(PATH_TO_MODEL)
    model = LitURNet3d.load_from_checkpoint(PATH_TO_MODEL)

    # trainer.tune(model)

    # Save the intermediate OrderedDict.
    trainer.fit(model)



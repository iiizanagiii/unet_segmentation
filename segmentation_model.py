from torch import nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss

class SegmentationModel(nn.Module):

  def __init__(self,ENCODER,WEIGHTS):
    super(SegmentationModel,self).__init__()

    self.arc = smp.Unet(
        encoder_name = ENCODER,
        encoder_weights =   WEIGHTS,
        in_channels = 3,
        classes = 1,
        activation = 'sigmoid'
    )

  def forward(self,images,masks=None):

    logits = self.arc(images)

    if masks != None:
      loss1 = DiceLoss(mode = 'binary')(logits,masks)
      loss2 = nn.BCEWithLogitsLoss()(logits,masks)
      return logits,loss1+loss2

    return logits
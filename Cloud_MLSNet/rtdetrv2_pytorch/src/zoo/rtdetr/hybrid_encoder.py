"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""


import torch 
import torch.nn as nn
from ...core import register
from ...nn.backbone.slimneck import *
from .aifi import *
from .adds import *


__all__ = ['Slim_neck_C3']

@register()
class Slim_neck_C3(nn.Module):
    def __init__(self,):
        super().__init__()
        self.gsconv_1 = GSConv(128,256,1,1,None,1,1,False)
        self.gsconv_2 = GSConv(256, 256, 1, 1, None, 1, 1, False)
        self.gsconv_3 = GSConv(512, 256, 1, 1, None, 1, 1, False)

        self.aifi = AIFI(256)
        self.gsconv_4 = GSConv(256, 256, 1, 1, None, 1, 1, False)
        self.upsample_1 = nn.Upsample(None, 2,'nearest')
        self.concat_1 = Concat(1)

        self.vovgscsp_1 = VoVGSCSP(512,256)
        self.vovgscsp_1_1 = VoVGSCSP(256,256)
        self.vovgscsp_1_2 = VoVGSCSP(256, 256)

        self.gsconv_5 = GSConv(256,256,1,1,None,1,1,False)
        self.upsample_2 = nn.Upsample(None, 2, 'nearest')
        self.concat_2 = Concat(1)

        ##开始输出到head
        self.vovgscsp_2 = VoVGSCSP(512,256)
        self.vovgscsp_2_1 = VoVGSCSP(256,256)
        self.vovgscsp_2_2 = VoVGSCSP(256, 256)

        self.gsconv_6 = GSConv(256,256,3,2)
        self.concat_3 = Concat(1)

        self.vovgscsp_3 = VoVGSCSP(512,256)
        self.vovgscsp_3_1 = VoVGSCSP(256,256)
        self.vovgscsp_3_2 = VoVGSCSP(256, 256)

        self.gsconv_7 = GSConv(256, 256,3,2)
        self.concat_4 = Concat(1)

        self.vovgscsp_4 = VoVGSCSP(512,256)
        self.vovgscsp_4_1 = VoVGSCSP(256,256)
        self.vovgscsp_4_2 = VoVGSCSP(256, 256)

        self.add1 = ADD_Block(256,0.25)
        self.add2 = ADD_Block(256,1)
        self.add3 = ADD_Block(256,4)

    def forward(self,x):
        x1, x2, x3 = x[0], x[1], x[2]

        x3_to_mix = self.gsconv_3(x3)
        x3 = self.aifi(x3_to_mix)
        x3_out = self.gsconv_4(x3) ##输出一下
        x3 = self.upsample_1(x3_out)

        x2_to_mix = self.gsconv_2(x2)
        x2_3= self.concat_1([x3,x2_to_mix])

        x2_3 = self.vovgscsp_1(x2_3)
        x2_3 = self.vovgscsp_1_1(x2_3)
        x2_3 = self.vovgscsp_1_2(x2_3)
        x2_3_out = self.gsconv_5(x2_3)  ##输出一下
        x2_3 = self.upsample_2(x2_3_out)

        x1_to_mix = self.gsconv_1(x1)
        x_total = self.concat_2([x2_3,x1_to_mix])

        x_total = self.vovgscsp_2(x_total)
        x_total = self.vovgscsp_2_1(x_total)
        x_out1 = self.vovgscsp_2_2(x_total) ##输出

        x_total = self.gsconv_6(x_out1)
        x_total = self.concat_3([x2_3_out,x_total])

        x_total = self.vovgscsp_3(x_total)
        x_total = self.vovgscsp_3_1(x_total)
        x_out2 = self.vovgscsp_3_2(x_total) ##输出

        x_total = self.gsconv_7(x_out2)
        x_total = self.concat_4([x3_out,x_total])

        x_total = self.vovgscsp_4(x_total)
        x_total = self.vovgscsp_4_1(x_total)
        x_out3 = self.vovgscsp_4_2(x_total) ##输出



        x_out1 = self.add1([x3_to_mix, x_out1])
        x_out2 = self.add2([x2_to_mix, x_out2])
        x_out3 = self.add3([x1_to_mix, x_out3])
        return (x_out1, x_out2, x_out3)


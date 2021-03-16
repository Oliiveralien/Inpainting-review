import torch
import torch.nn as nn
import torch.nn.functional as F

class StyleLoss(nn.Module):
    def __init__(self, target, patch_size, mrf_style_stride, mrf_synthesis_stride, gpu_chunck_size, device):
        super(StyleLoss, self).__init__()
        self.patch_size = patch_size
        self.mrf_style_stride = mrf_style_stride
        self.mrf_synthesis_stride = mrf_synthesis_stride
        self.gpu_chunck_size = gpu_chunck_size
        self.device = device
        self.loss = None

        self.style_patches = self.patches_sampling(target.detach(), patch_size=self.patch_size, stride=self.mrf_style_stride)
        self.style_patches_norm = self.cal_patches_norm()
        self.style_patches_norm = self.style_patches_norm.view(-1, 1, 1)

    def update(self,target):
        self.style_patches=self.patches_sampling(target.detach(),patch_size=self.patch_size,stride=self.mrf_style_stride)
        self.style_patches_norm=self.cal_patches_norm()
        self.style_patches_norm=self.style_patches_norm.view(-1,1,1)

    def forward(self, input):
        sysnthesis_patches=self.content_patches_sampling(input,patch_size=self.patch_size,stride=self.mrf_synthesis_stride)
        max_response=[]
        for i in range(0, self.style_patches.shape[0], self.gpu_chunck_size):
            i_start = i
            i_end = min(i + self.gpu_chunck_size, self.style_patches.shape[0])
            weight = self.style_patches[i_start:i_end, :, :, :]
            response = F.conv2d(input, weight, stride=self.mrf_synthesis_stride)
            max_response.append(response.squeeze(dim=0))
        max_response = torch.cat(max_response, dim=0)

        max_response = max_response.div(self.style_patches_norm)
        max_response = torch.argmax(max_response, dim=0)
        max_response = torch.reshape(max_response, (1, -1)).squeeze()

        loss=0
        for i in range(0,len(max_response),self.gpu_chunck_size):
            i_start=i
            i_end=min(i+self.gpu_chunck_size,len(max_response))
            tp_ind=tuple(range(i_start,i_end))
            sp_ind=max_response[i_start:i_end]
            loss+=torch.sum(torch.mean(torch.pow(sysnthesis_patches[tp_ind,:,:,:]-self.style_patches[sp_ind,:,:,:],2),dim=[1,2,3]))
        self.loss=loss/len(max_response)
        return input





    #提取出所有的patches
    def patches_sampling(self,image,patch_size,stride):
        h,w=image.shape[2:4]
        patches=[]
        for i in range(0,h-patch_size+1,stride):
            for j in range(0,w-patch_size+1,stride):
                centerX=i+self.patch_size/2
                centerY=j+self.patch_size/2
                bool=(centerX>h/4)and(centerX<(h*3/4))and(centerY>w/4)and(centerY<(w*3/4))
                if(not bool):
                    patches.append(image[:,:,i:i+patch_size,j:j+patch_size])
        patches=torch.cat(patches,dim=0).to(self.device)
        return patches



    def content_patches_sampling(self, image, patch_size, stride):
        h, w = image.shape[2:4]
        patches = []
        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                patches.append(image[:, :, i:i + patch_size, j:j + patch_size])
        patches = torch.cat(patches, dim=0).to(self.device)
        return patches


    #计算每张图片的norm
    def cal_patches_norm(self):
        norm_array=torch.zeros(self.style_patches.shape[0])
        for i in range(self.style_patches.shape[0]):
            norm_array[i] = torch.pow(torch.sum(torch.pow(self.style_patches[i], 2)), 0.5)
        return norm_array.to(self.device)

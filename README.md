# DescRetinaFace 

This repo is largely based on the following RetineFace implementation:
https://github.com/biubug6/Pytorch_Retinaface

With the following modifications:
- remove per scale per task head
- single descriptor or ms-descriptor head to allow for per scale desc + per task hear or single desc across scales + per task 
- triplet loss
-  attention before conv

## WiderFace Val Performance in single scale When using Resnet50 as backbone net and original image scale 
| Arch | Num Epochs | easy | medium | hard |
|:-|:-:|:-:|:-:|:-:|
| RetinaFace (Ref Pytorch Impl.) | 100 | xx% | xx% | xx% |
| RetinaFace (Ref Pytorch Impl.) | 10 | 91.39% | 89.54% | 76.85% |
| DescRetinaFace w MS Desc Head| 10 | xx% | xx% | xx% |
| DescRetinaFace w SS Desc | 10 | xx% | xx% | xx% |`
| DescRetinaFace w MS Desc Head w Triplet Loss (bs=4)| 10 | xx% | xx% | xx% |
| DescRetinaFace w MS Desc Head w Sparse Attention| 10 | xx% | xx% | xx% |




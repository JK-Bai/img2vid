# Image_to_video
本项目使用了`stable-video-diffusion-img2vid-xt-1-1`预训练模型
## 如何使用
1，`pip install -r requirements.txt` <br>
2，替换项目中的`pic.png`为您的输入图片 <br>
3，Run：`python main.py` <br>
## 可能遇到的问题及解决方案
问题：<br>
无法访问受限模型 <br>
解决方案：<br>
1，登录 Hugging Face 账号<br>
2，登录后，进入 Hugging Face Token 页面，生成一个新的访问令牌。建议选择 "Write" 级别的权限，复制您的Token。<br>
3，在终端中运行以下命令来确认你已正确登录 Hugging Face CLI：`huggingface-cli login`，这里需要输入Token。


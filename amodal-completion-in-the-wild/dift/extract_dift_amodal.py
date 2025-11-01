import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import argparse
import torch
from PIL import Image
from torchvision.transforms import PILToTensor
from dift.dift_sd import SDFeaturizer
import os
import json
import ipdb

def main(args):
    dift = SDFeaturizer(args.model_id)
    img = Image.open(args.input_path).convert('RGB')
    if args.img_size[0] > 0:
        img = img.resize(args.img_size)
    img_tensor = (PILToTensor()(img) / 255.0 - 0.5) * 2
    ft = dift.forward(img_tensor,
                      prompt=args.prompt,
                      t=args.t,
                      up_ft_index=args.up_ft_index,
                      ensemble_size=args.ensemble_size)
    return ft


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='''extract dift from input image, and save it as torch tenosr,
                    in the shape of [c, h, w].''')
    
    parser.add_argument('--img_size', nargs='+', type=int, default=[768, 768],
                        help='''in the order of [width, height], resize input image
                            to [w, h] before fed into diffusion model, if set to 0, will
                            stick to the original input size. by default is 768x768.''')
    parser.add_argument('--model_id', default='stabilityai/stable-diffusion-2-1', type=str, 
                        help='model_id of the diffusion model in huggingface')
    parser.add_argument('--t', default=261, type=int, 
                        help='time step for diffusion, choose from range [0, 1000]')
    parser.add_argument('--up_ft_index', default=1, type=int, choices=[0, 1, 2 ,3],
                        help='which upsampling block of U-Net to extract the feature map')
    parser.add_argument('--prompt', default='', type=str,
                        help='prompt used in the stable diffusion')
    parser.add_argument('--ensemble_size', default=8, type=int, 
                        help='number of repeated images in each batch used to get features')
    parser.add_argument('--input_path', type=str,
                        help='path to the input image file')
    parser.add_argument('--output_path', type=str, default='dift.pt',
                        help='path to save the output features as torch tensor')
    args = parser.parse_args()

    t = 181
    print(t)


    # img_dir = '' # fill in the path to the image directory
    save_dir = '/data1/CLB/amodal-completion-in-the-wild/kins_sd_features' # fill in the path to the directory of saving the extracted features
    # 2. 设置包含图像路径列表的JSON文件路径。
    #  需要为训练集和测试集分别运行此脚本。
    #  替换下面的占位符路径。
    img_list_file = '/data1/CLB/amodal-completion-in-the-wild/kins_train_images.json'
    # 第二次运行时，请将上面一行修改为:
    # img_list_file = '/data1/CLB/amodal-completion-in-the-wild/kins_test_images.json'
    # 废弃img_name_list = json.load(open('')) # fill in the path to the image name list
    # 从JSON文件中加载完整的图像路径列表。
    img_path_list = json.load(open(img_list_file))
    print(f"正在处理来自 {img_list_file} 的 {len(img_path_list)} 张图像...")

    # 修改主循环，以处理完整的图像路径列表。
   # --- 修改后的循环逻辑 ---
    for i, img_path in enumerate(img_path_list):
        img_basename = os.path.basename(img_path)
        output_filename = img_basename.replace('.png', '.pt')

        # <<<--- 新增的检查逻辑 --- START
        # 我们检查最浅层(up_ft_index=3)的特征文件是否存在作为代表
        check_folder = os.path.join(save_dir, f't_{t}_up-ft-index_3')
        check_file_path = os.path.join(check_folder, output_filename)

        if os.path.exists(check_file_path):
            print(f"  - ({i+1}/{len(img_path_list)}) [已跳过] 特征文件已存在: {img_basename}")
            continue # 如果文件存在，就跳过这张图片
        # <<<--- 新增的检查逻辑 --- END

        print(f"  - ({i+1}/{len(img_path_list)}) [正在处理] {img_basename}")
        args.input_path = img_path
        args.prompt = ''

        try:
            ft = main(args)

            for key_i in ft.keys():
                cur_folder = os.path.join(save_dir, f't_{t}_up-ft-index_{key_i}')
                os.makedirs(cur_folder, exist_ok=True)
                args.output_path = os.path.join(cur_folder, output_filename)
                torch.save(ft[key_i].squeeze(0).cpu(), args.output_path)
        except Exception as e:
            print(f"!!!!!! 处理 {img_basename} 时发生错误: {e}")
            continue

    print("特征提取完成。")
    """ for img_name in img_name_list:
        print(img_name)
        args.t = t
        args.up_ft_index = 1
        args.input_path = os.path.join(img_dir, img_name)
        args.prompt = ''
        ft = main(args)
        for key_i in ft.keys():
            cur_folder = os.path.join(save_dir, 't_' + str(t) + '_up-ft-index_' + str(key_i))
            if not os.path.exists(cur_folder):
                os.mkdir(cur_folder)
            args.output_path = os.path.join(cur_folder, img_name[:-4] + '.pt')
            cur_ft = torch.save(ft[key_i].squeeze(0).cpu(), args.output_path) """

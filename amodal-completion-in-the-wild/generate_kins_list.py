import os
import json

# --- 请确保修改为您的实际路径 ---
kins_root_path = '/data1/CLB/KINS' 
output_dir = '.' 
# --------------------------------

def create_image_list(dataset_split):
    # 处理 KINS 数据集 'training' 和 'testing' 文件夹的命名差异
    folder_name = 'testing' if dataset_split == 'test' else f'{dataset_split}ing'
    
    image_dir = os.path.join(kins_root_path, folder_name, 'image_2')
    
    if not os.path.exists(image_dir):
        print(f"Directory not found: {image_dir}")
        return

    # 确保只包含以 '.png' 结尾的文件
    image_files = [os.path.join(image_dir, f) for f in sorted(os.listdir(image_dir)) if f.endswith('.png')]

    output_filename = os.path.join(output_dir, f'kins_{dataset_split}_images.json')
    with open(output_filename, 'w') as f:
        json.dump(image_files, f, indent=4)
        
    print(f"Generated {len(image_files)} image paths in '{output_filename}'")

# 为训练集和测试集分别生成列表
create_image_list('train')
create_image_list('test')
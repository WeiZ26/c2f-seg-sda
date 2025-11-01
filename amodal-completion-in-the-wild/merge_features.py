import os
import torch
from tqdm import tqdm
import json

# --- 请修改以下路径 ---
# 原始 SD 特征所在的根目录
source_feature_dir = "/data1/CLB/amodal-completion-in-the-wild/kins_sd_features origin"
# 您希望保存合并后特征的新目录
target_feature_dir = "/data1/CLB/amodal-completion-in-the-wild/kins_sd_features"
# 您之前为特征提取创建的图像列表
image_list_json = "/data1/CLB/amodal-completion-in-the-wild/kins_train_images.json"
# ----------------------

# --- 新增：选择要合并和删除的特征层级 ---
# 我们只需要 0 到 3 这四个层级
feature_indices_to_process = range(4) 
# ------------------------------------

def merge_and_delete_features():
    # 确保目标目录存在，以便进行检查
    os.makedirs(target_feature_dir, exist_ok=True)
    
    with open(image_list_json, 'r') as f:
        image_paths = json.load(f)

    print(f"开始合并 {len(image_paths)} 个图像的特征，并将在合并后删除原始文件...")
    print("脚本将自动跳过已合并的文件。")

    for img_path in tqdm(image_paths, desc="合并进度"):
        image_basename = os.path.basename(img_path).replace('.png', '.pt')
        target_file_path = os.path.join(target_feature_dir, image_basename)
        
        # <<<--- 新增的断点续传检查逻辑 --- START
        # 如果合并后的文件已经存在，就直接跳过
        if os.path.exists(target_file_path):
            continue
        # <<<--- 新增的断点续传检查逻辑 --- END
            
        original_feature_files = []
        features_to_merge = []
        
        try:
            # 1. 收集所有原始文件的路径并加载它们
            for i in feature_indices_to_process:
                feature_file = os.path.join(source_feature_dir, f't_181_up-ft-index_{i}', image_basename)
                
                if not os.path.exists(feature_file):
                    raise FileNotFoundError(f"找不到文件: {feature_file}")

                original_feature_files.append(feature_file)
                feature = torch.load(feature_file, map_location='cpu')
                features_to_merge.append(feature)
            
            # 2. 保存合并后的文件
            torch.save(features_to_merge, target_file_path)

            # 3. 只有在确认新文件已成功保存后，才开始删除原始文件
            for file_to_delete in original_feature_files:
                os.remove(file_to_delete)

        except FileNotFoundError as e:
            # 这个错误现在应该很少见了，但保留以策万全
            print(f"\n警告：{e}。已跳过 {image_basename}。")
            continue
        except Exception as e:
            print(f"\n处理 {image_basename} 时发生未知错误: {e}。已跳过，未删除文件。")
            continue

    print("所有特征合并和删除操作完成！")

if __name__ == '__main__':
    # 删除了之前的用户确认步骤，因为现在脚本是安全的断点续传模式
    merge_and_delete_features()
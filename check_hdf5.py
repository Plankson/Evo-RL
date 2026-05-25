import os
import h5py
import numpy as np

def print_hdf5_structure(file_path):
    """
    读取并打印 HDF5 文件的全部 Key 和相关信息
    """
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在 -> {file_path}")
        return

    print(f"正在读取文件: {file_path}")
    print("=" * 60)

    try:
        with h5py.File(file_path, 'r') as f:
            def visitor_func(name, node):
                # 计算缩进以展示层级结构
                indent = "  " * name.count('/')
                
                if isinstance(node, h5py.Group):
                    print(f"{indent}📁 Group: {name}")
                elif isinstance(node, h5py.Dataset):
                    print(f"{indent}📄 Dataset: {name}")
                    print(f"{indent}   - Shape: {node.shape}")
                    print(f"{indent}   - Type : {node.dtype}")
                    # 如果数据集很小，可以选择打印它的前几个数据
                    if node.size <= 5:
                        print(f"{indent}   - Value: {node[:]}")
                    elif len(node.shape) > 0 and node.shape[0] > 0:
                        print(f"{indent}   - Length: {node.shape[0]}")
                print("-" * 40)

            # visititems 会递归遍历文件中的所有对象（Group 和 Dataset）
            f.visititems(visitor_func)
            
            # 检查 complementary_info/pred_cluster_idx
            print("=" * 60)
            pred_cluster_idx_key = "complementary_info/pred_cluster_idx"
            if pred_cluster_idx_key in f:
                pred_cluster_idx = np.asarray(f[pred_cluster_idx_key][:]).reshape(-1)
                print(f"✓ 找到 {pred_cluster_idx_key}")
                print(f"  - 长度: {len(pred_cluster_idx)}")
                print(f"  - 数据类型: {pred_cluster_idx.dtype}")
                
                # 检查是否所有值都一致
                unique_values = np.unique(pred_cluster_idx)
                if len(unique_values) == 1:
                    print(f"  ✓ 所有值一致: {unique_values[0]}")
                else:
                    print(f"  ✗ 值不一致，共有 {len(unique_values)} 种不同的值:")
                    for val in unique_values:
                        count = np.sum(pred_cluster_idx == val)
                        print(f"     - 值 {val}: 出现 {count} 次")
            else:
                print(f"✗ 未找到 {pred_cluster_idx_key}")
            
    except Exception as e:
        print(f"读取 HDF5 文件时发生错误: {e}")

if __name__ == "__main__":
    # 指定你的文件路径
    target_file = "/home/agilex/evorl_dataset/pi0/0518/fold_clothes_advantage_positive_new_ck/fail/episode_13.hdf5"
    print_hdf5_structure(target_file)
import numpy as np
import scipy.io as sio
from sklearn.metrics import confusion_matrix, cohen_kappa_score

pre_path = './pavia_dual0.mat'
# gt_path = './pavia/paviaU_boundary_only.mat'
# gt_path = './pavia/paviaU_boundary_only.mat'
gt_path = './pavia/paviaU_gt10.mat'
# gt_path = './indianpines/indian_gt9.mat'
# gt_path = './indianpines/indian_pines_edges.mat'
# gt_path = './salinas/salinas_gt17.mat'
# gt_path = './salinas/processed_salinas_gt17.mat'
pre_data = sio.loadmat(pre_path)
gt_data = sio.loadmat(gt_path)

# Indian Pines
pre = pre_data['prob']
gt = gt_data['paviaU']
# gt = gt_data['paviaU_boundary_only']
# gt = gt_data['indian_pines']
labels = [1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17]  # Indian Pines

# 初始化计数器
total_correct = 0
total_samples = 0
acc_per_class = []

# 计算每个类别的准确率
for k in labels:
    a = (pre == k)
    b = (gt == k)
    correct = np.sum(a & b)
    total = np.sum(b)
    if total > 0:
        acc = correct / total
        acc_per_class.append(acc)
        print(f"Class {k} Acc: {acc:.4f}")
    total_correct += correct
    total_samples += total

# 计算OA
oa = total_correct / total_samples
print(f"Overall Acc (OA): {oa:.4f}")

# 计算AA
aa = np.mean(acc_per_class)
print(f"Average Acc (AA): {aa:.3f}")

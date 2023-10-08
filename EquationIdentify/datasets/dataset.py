import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence

def load_lbl2id_map(lbl2id_map_path):
    """
    读取 字符-id 映射关系记录的txt文件，并返回 lbl->id 和 id->lbl 映射字典
    lbl2id_map_path : 字符-id 映射关系记录的txt文件路径
    """
    lbl2id_map = dict()
    id2lbl_map = dict()
    with open(lbl2id_map_path, 'r', encoding='utf-8') as reader:
        for line in reader:
            label, cur_id = line.rstrip().split('\t')
            lbl2id_map[label] = int(cur_id)
            id2lbl_map[int(cur_id)] = label

    return lbl2id_map, id2lbl_map


当然可以优化一下。以下是一个优化版本，包括函数名和代码结构的改进：

python
Copy code
def generate_label_id_mapping(label_id_map_path, label_count_map):
    """
    生成字符到ID的映射并保存到文件
    :param label_id_map_path: 文件保存路径
    :param label_count_map: 记录标签中字符出现次数的字典
    """
    # 构造label中字符--id之间的映射
    label_id_map = {'☯': 0, '■': 1, '□': 2}  # 初始化三个特殊字符

    # 生成其余字符的ID映射关系
    cur_id = 3
    for label in label_count_map.keys():
        label_id_map[label] = cur_id
        cur_id += 1

    # 保存字符--ID之间的映射到txt文件
    with open(label_id_map_path, 'w', encoding='utf-8') as writer:
        for label, cur_id in label_id_map.items():
            line = f"{label}\t{cur_id}\n"
            writer.write(line)



class FormulaDataset(Dataset):
    def __init__(self, data_folder, lbl2id_map_path, tokenizer, transform=None):
        self.data_folder = data_folder
        self.image_folder = os.path.join(data_folder, 'images')
        self.latex_folder = os.path.join(data_folder, 'latex')
        self.image_list = os.listdir(self.image_folder)
        self.latex_list = os.listdir(self.latex_folder)
        self.transform = transform
        self.lbl2id_map_path = lbl2id_map_path
        self.lbl2id_map, self.id2lbl_map = self.load_lbl2id_map()
        self.tokenizer = tokenizer

    def load_lbl2id_map(self):
        lbl2id_map, id2lbl_map = load_lbl2id_map(self.lbl2id_map_path)
        return lbl2id_map, id2lbl_map

    def latex_to_tensor(self, latex_code):
        tokens = self.tokenizer.tokenize(latex_code)
        return torch.tensor([self.lbl2id_map[token] for token in tokens], dtype=torch.long)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        # 读取图片
        img_name = os.path.join(self.image_folder, self.image_list[idx])
        image = Image.open(img_name).convert('RGB')

        # 读取 LaTeX 文件
        latex_name = os.path.join(self.latex_folder, self.latex_list[idx])
        with open(latex_name, 'r') as file:
            latex_code = file.read()

        # LaTeX 代码分词
        latex_tensor = self.latex_to_tensor(latex_code)

        # 图像转换
        if self.transform:
            image = self.transform(image)

        return image, latex_tensor

# 数据集和数据加载器
data_folder = 'path/to/your/data'
lbl2id_map_path = 'path/to/your/lbl2id_map.txt'
# 生成字符到ID的映射并保存到文件
generate_label_id_mapping(label_id_map_path, label_count_map)
# 然后将lbl2id_map, id2lbl_map传递给FormulaDataset
formula_dataset = FormulaDataset(data_folder, lbl2id_map_path, transform=transform)
data_loader = DataLoader(formula_dataset, batch_size=32, shuffle=True, collate_fn=lambda batch: pad_sequence(batch, batch_first=True))
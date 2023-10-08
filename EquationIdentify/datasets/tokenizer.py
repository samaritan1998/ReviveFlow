class Tokenizer:
    def __init__(self, method='fmm', user_dict=None):
        """
        初始化分词器
        :param method: 分词方法，支持 'fmm'（正向最大匹配）和其他方法
        :param user_dict: 用户自定义词典
        """
        self.method = method
        self.user_dict = user_dict

    def fmm_func(self, sentence):
        """
        正向最大匹配（FMM）
        :param sentence: 句子
        :return: 分词结果列表
        """
        if self.user_dict is None:
            raise ValueError("用户词典不能为空")

        max_len = max([len(item) for item in self.user_dict])
        start = 0
        token_list = []
        while start != len(sentence):
            index = start + max_len
            if index > len(sentence):
                index = len(sentence)
            for i in range(max_len):
                if (sentence[start:index] in self.user_dict) or (len(sentence[start:index]) == 1):
                    token_list.append(sentence[start:index])
                    start = index
                    break
                index += -1

        return token_list

    def tokenize(self, sentence):
        """
        对句子进行分词
        :param sentence: 句子
        :return: 分词结果列表
        """
        if self.method == 'fmm':
            return self.fmm_func(sentence)
        else:
            raise ValueError("不支持的分词方法")

# 使用示例
user_dict = {'我', '爱', '自然语言处理', '自然', '语言', '处理'}
tokenizer = Tokenizer(method='fmm', user_dict=user_dict)
sentence = '我爱自然语言处理'
result = tokenizer.tokenize(sentence)
print(result)

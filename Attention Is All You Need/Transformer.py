import math

import torch
import torch.optim as optim
import torch.nn as nn



class Transformer(nn.Module):
    """
    my implementation of Transformer
    """
    def __init__(self, d_model: int = 512, nhead: int = 8, dk: int = 64, dv: int = 64, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_ffn: int = 2048, src_vocabulary_size: int = 3000,
                 tgt_vocabulary_size: int = 3000, max_len: int = 5000, dropout: float = 0.1):
        """
        创建Transformer模型
        :param d_model: 词向量的维度
        :param nhead: 多头注意力中头的数量
        :param dk: query和key的向量维度
        :param dv: value的向量维度
        :param num_encoder_layers: encoder layer的层数
        :param num_decoder_layers: decoder layer的层数
        :param dim_ffn: 前馈全连接网络的中间神经元个数
        :param src_vocabulary_size: src单词表大小，用于word embedding
        :param tgt_vocabulary_size: tgt单词表大小，用于word embedding
        :param max_len: 要大于src和tgt中最长序列的长度，用于position embedding
        :param dropout: dropout的drop probability
        """
        super(Transformer, self).__init__()

        # 创建encoder模块
        self.encoder = Encoder(d_model, nhead, dk, dv, num_encoder_layers, dim_ffn, src_vocabulary_size, max_len, dropout)
        # 创建decoder模块
        self.decoder = Decoder(d_model, nhead, dk, dv, num_decoder_layers, dim_ffn, tgt_vocabulary_size, max_len, dropout)
        # 创建将词向量映射到类别的projection模块
        self.projection = nn.Linear(d_model, tgt_vocabulary_size, bias=False)


    def forward(self, encoder_input, decoder_input, self_attn_mask, masked_self_attn_mask, cross_self_attn_mask):
        """
        :param encoder_input: [batch_size, src_len]
        :param decoder_input: [batch_size, tgt_len]
        :param self_attn_mask: 编码器中自注意力的mask
        :param masked_self_attn_mask: 解码器中masked自注意力的mask
        :param cross_self_attn_mask: 解码器中cross自主注意力的mask
        :return:
        """

        encoder_output, encoder_self_attn_scores = self.encoder(encoder_input, self_attn_mask)
        # encoder_output为编码器的输出memory，之后用于指导解码器的解码工作，[batch_size, src_len, d_model]
        # encoder_self_attn_scores为编码器各个EncoderLayer中self-attention模块得到的注意力分数，用于可视化，[num_encoder_layers, batch_size, nhead, src_len, src_len]

        decoder_output, decoder_masked_self_attn_scores, decoder_cross_self_attn_scores = \
            self.decoder(decoder_input, encoder_output, masked_self_attn_mask, cross_self_attn_mask)
        # decoder_output为解码器的输出，之后会输入到映射分类模块，[batch_size, tgt_len, d_model]
        # decoder_masked_self_attn_scores为解码器各个DecoderLayer中masked self-attention模块得到的注意力分数，用于可视化，[num_decoder_layers, batch_size, nhead, tgt_len, tgt_len]
        # decoder_cross_self_attn_scores为解码器各个DecoderLayer中cross self-attention模块得到的注意力分数，用于可视化，[num_decoder_layers, batch_size, nhead, tgt_len, src_len]

        decoder_logits = self.projection(decoder_output)
        # decoder_logits为序列中各个token的类别概率，[batch_size, tgt_len, tgt_vocabulary_size]

        return decoder_logits, encoder_self_attn_scores, decoder_masked_self_attn_scores, decoder_cross_self_attn_scores



# 1
class Encoder(nn.Module):
    """
    my implementation of Encoder
    """
    def __init__(self, d_model: int = 512, nhead: int = 8, dk: int = 64, dv: int = 64, num_encoder_layers: int = 6,
                 dim_ffn: int = 2048, src_vocabulary_size: int = 3000, max_len: int = 5000, dropout: float = 0.1):
        """
        创建Encoder模型
        :param d_model: 词向量的维度
        :param nhead: 多头注意力中头的数量
        :param dk: query和key的向量维度
        :param dv: value的向量维度
        :param num_encoder_layers: encoder layer的层数
        :param dim_ffn: 前馈全连接网络的中间神经元个数
        :param src_vocabulary_size: src单词表大小，用于word embedding
        :param max_len: 要大于src和tgt中最长序列的长度，用于position embedding
        :param dropout: dropout的drop probability
        """
        super(Encoder, self).__init__()

        # 定义word_embedding
        self.src_we = nn.Embedding(src_vocabulary_size, d_model)
        for param in self.src_we.parameters():
            print(param)
            print(param.shape)

        # 定义position_embedding
        self.src_pe = PositionEmbedding(max_len, d_model, dropout)

        # 定义encoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, nhead, dk, dv, dim_ffn, dropout) for _ in range(num_encoder_layers)
        ])


    def forward(self, encoder_input, self_attn_mask):
        # encoder_input: [batch_size, src_len]
        # self_attn_mask: [batch_size, src_len, src_len]

        # 词嵌入word embedding
        encoder_seq = self.src_we(encoder_input)  # [batch_size, src_len, d_model]

        # 添加位置信息
        encoder_seq = self.src_pe(encoder_seq)  # [batch_size, src_len, d_model]

        # 经过数个EncoderLayer
        encoder_self_attn_scores = []
        for encoder_layer in self.encoder_layers:
            # 调用EncoderLayer
            encoder_seq, score = encoder_layer(encoder_seq, self_attn_mask)  # [batch_size, src_len, d_model]
            # 将每层的自注意力分数存起来
            encoder_self_attn_scores.append(score)

        return encoder_seq, torch.stack(encoder_self_attn_scores, dim=0)



# 1.1
class PositionEmbedding(nn.Module):
    """
    位置向量添加器
    """
    def __init__(self, max_len: int = 5000, d_model: int = 512, dropout: float = 0.1):
        """
        将位置向量生成器创建为nn.Embedding
        :param max_len: 序列可能的最大长度
        :param d_model: 词向量维度
        :param dropout: dropout的drop probability
        """
        super(PositionEmbedding, self).__init__()

        # 先定义一个(max_len,1)的张量存储行的值，表示pos的递增
        pos_mat = torch.arange(max_len).reshape((-1, 1))
        # 再定义一个(1,d_model/2)的张量存储维度的值，表示2i的递增
        i_mat = torch.arange(0, d_model, 2).reshape((1, -1)) / d_model
        # i_mat计算以10000为底的次方
        i_mat = torch.pow(10000.0, i_mat)

        # 定义一个position_embedding表，大小为[max_len, d_model]
        pe_table = torch.zeros(max_len, d_model)
        # 根据列的奇偶分别计算sin和cos函数并进行赋值
        pe_table[:, 0::2] = torch.sin(pos_mat / i_mat)
        pe_table[:, 1::2] = torch.cos(pos_mat / i_mat)

        # 实例化nn.Embedding
        self.pe = nn.Embedding(max_len, d_model)
        # 将pe_table传入作为它的weight
        self.pe.weight = nn.Parameter(pe_table, requires_grad=False)

        # 定义输出之前的Dropout
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, x):
        # x:[batch_size, src_len/tgt_len, d_model]

        # 获取x的维度信息
        N, seq_len, model_dim = x.size()
        # 根据维度信息生成x的位置张量
        pos_x = torch.stack([torch.arange(seq_len) for _ in range(N)], dim=0)  # [batch_size, seq_len]

        # 通过self.pe计算得到对应词向量的位置向量
        pe_x = self.pe(pos_x) # [batch_size, seq_len, d_model]

        return self.dropout(x + pe_x)



# 1.2
class EncoderLayer(nn.Module):
    """
    my implementation of EncoderLayer
    """
    def __init__(self, d_model: int = 512, nhead: int = 8, dk: int = 64, dv: int = 64,
                 dim_ffn: int = 2048, dropout: float = 0.1):
        """
        创建编码器层
        :param d_model: 词向量的维度
        :param nhead: 多头注意力中头的数量
        :param dk: query和key的向量维度
        :param dv: value的向量维度
        :param dim_ffn: 前馈全连接网络的中间神经元个数
        :param dropout: dropout的drop probability
        """
        super(EncoderLayer, self).__init__()

        # 初始化自注意力模块
        self.encoder_self_attn = MultiHeadSelfAttention(d_model, nhead, dk, dv, dropout)

        # 初始化FFN(feedforward network)模块
        self.encoder_ffn = PositionwiseFeedForwardNetwork(d_model, dim_ffn, dropout)


    def forward(self, encoder_seq, self_attn_mask):
        # encoder_seq: [batch_size, src_len, d_model]
        # self_attn_mask: [batch_size, src_len, src_len]

        # 经过自注意力模块
        encoder_inner_seq, score = self.encoder_self_attn(encoder_seq, encoder_seq, encoder_seq, self_attn_mask)
        # encoder_inner_seq: [batch_size, src_len, d_model]
        # score: [batch_size, nhead, src_len, src_len]

        # 经过FFN模块
        return self.encoder_ffn(encoder_inner_seq), score



# 1.2.1
class MultiHeadSelfAttention(nn.Module):
    """
    my implementation of MultiHeadSelfAttention
    """
    def __init__(self, d_model,  nhead, dk, dv, dropout):
        """
        创建多头注意力模块
        :param d_model: 词向量的维度
        :param nhead: 多头注意力中头的数量
        :param dk: query和key的向量维度
        :param dropout: dropout的drop probability
        """
        super(MultiHeadSelfAttention, self).__init__()
        # 保存一些要用的参数
        self.nhead = nhead
        self.dk = dk
        self.dv = dv

        # 定义W_Q, W_K, W_V
        self.W_Q = nn.Linear(d_model, nhead * dk)
        self.W_K = nn.Linear(d_model, nhead * dk)
        self.W_V = nn.Linear(d_model, nhead * dv)
        # 定义将nhead*dv维度重新转换成d_model维度的全连接层
        self.linear = nn.Linear(nhead * dv, d_model)
        # 定义Dropout
        self.dropout = nn.Dropout(p=dropout)
        # 定义LayerNorm
        self.layer_norm = nn.LayerNorm(d_model)


    def forward(self, I_Q, I_K, I_V, attn_mask):
        # I_Q、I_K、I_V:[batch_size, seq_len(src_len/tgt_len), d_model]
        # attn_mask: [batch_size, seq_len(src_len/tgt_len), seq_len(src_len/tgt_len)]

        # 预先保留一份残差块的直传数据，并且获得batch_size
        residual, batch_size = I_Q, I_Q.size(0)

        # 先分别计算出Q、K、V，并且变换维度
        Q = self.W_Q(I_Q).view(batch_size, -1, self.nhead, self.dk).transpose(1, 2)  # [batch_size, nhead, seq_len, dk]
        K = self.W_K(I_K).view(batch_size, -1, self.nhead, self.dk).transpose(1, 2)  # [batch_size, nhead, seq_len, dk]
        V = self.W_V(I_V).view(batch_size, -1, self.nhead, self.dv).transpose(1, 2)  # [batch_size, nhead, seq_len, dv]

        # 将attn_mask张量在多头维度扩维
        attn_mask = attn_mask.unsqueeze(dim=1).repeat(1, self.nhead, 1, 1)  # [batch_size, nhead, seq_len, seq_len]

        # Scaled Dot-Product Attention
        context, score = ScaledDotProductAttention()(Q, K, V, attn_mask, self.dk)  # context:[batch_size, nhead, seq_len, dv]

        # 维度变换回来
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.nhead * self.dv)  # [batch_size, seq_len, nhead * dv]

        # 经过一个全连接层，向量维度由nhead*dv到d_model
        output = self.linear(context)  # [batch_size, seq_len, d_model]

        # 依次经过Dropout、Residual、LayerNorm
        output = self.layer_norm(self.dropout(output) + residual)

        return output, score



# 1.2.1.1
class ScaledDotProductAttention(nn.Module):
    """
    带有方差缩小的张量点成注意力
    """
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask, dk):
        # Q,K: [batch_size, nhead, seq_len, dk]
        # V:   [batch_size ,nhead, seq_len, dv]
        # attn_mask: [batch_size, nhead, seq_len, seq_len]

        # 计算(Q*K^T)/根号dk
        attn_score = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(torch.tensor(dk))  # [batch_size, nhead, seq_len, seq_len]

        # 进行mask
        attn_score.masked_fill_(attn_mask, -1e9)  # [batch_size, nhead, seq_len, seq_len]
        # 每行进行softmax
        attn_score = nn.Softmax(dim=-1)(attn_score)  # [batch_size, nhead, seq_len, seq_len]
        # 注意力权重张量与v相乘
        context = torch.matmul(attn_score, V)  # [batch_size, nhead, seq_len, dv]

        return context, attn_score



# 1.2.2
class PositionwiseFeedForwardNetwork(nn.Module):
    """
    my implementation of FFN
    """
    def __init__(self, d_model, dim_ffn, dropout):
        """
        创建position-wise前馈网络模块
        :param d_model: 词向量的维度
        :param dim_ffn: 前馈全连接网络的中间神经元个数
        :param dropout: dropout的drop probability
        """
        super(PositionwiseFeedForwardNetwork, self).__init__()

        # 初始化第一个全连接层
        self.linear1 = nn.Linear(d_model, dim_ffn)
        # 初始化激活函数
        self.relu = nn.ReLU()
        # 初始化第一个Dropout
        self.dropout1 = nn.Dropout(p=dropout)

        # 初始化第二个全连接层
        self.linear2 = nn.Linear(dim_ffn, d_model)
        # 初始化第二个Dropout
        self.dropout2 = nn.Dropout(p=dropout)

        # 初始化LayerNorm
        self.layer_norm = nn.LayerNorm(d_model)


    def forward(self, x):
        # x: [batch_size, seq_len, d_model]

        # 残差结构直接传递的数据
        residual = x  # [batch_size, seq_len, d_model]

        # Feed Forward
        output = self.dropout2(self.linear2(self.dropout1(self.relu(self.linear1(x)))))  # [batch_size, seq_len, d_model]

        # 残差连接，LayerNorm
        return self.layer_norm(output + residual)  # [batch_size, seq_len, d_model]



# 2
class Decoder(nn.Module):
    """
    my implementation of Decoder
    """
    def __init__(self, d_model: int = 512, nhead: int = 8, dk: int = 64, dv: int = 64, num_decoder_layers: int = 6,
                 dim_ffn: int = 2048, tgt_vocabulary_size: int = 3000, max_len: int = 5000, dropout: float = 0.1):
        """
        创建Decoder模型
        :param d_model: 词向量的维度
        :param nhead: 多头注意力中头的数量
        :param dk: query和key的向量维度
        :param dv: value的向量维度
        :param num_decoder_layers: decoder layer的层数
        :param dim_ffn: 前馈全连接网络的中间神经元个数
        :param tgt_vocabulary_size: tgt单词表大小，用于word embedding
        :param max_len: 要大于src和tgt中最长序列的长度，用于position embedding
        :param dropout: dropout的drop probability
        """
        super(Decoder, self).__init__()

        # 定义word_embedding
        self.decoder_we = nn.Embedding(tgt_vocabulary_size, d_model)

        # 定义position_embedding
        self.decoder_pe = PositionEmbedding(max_len, d_model, dropout)

        # 定义decoder layers
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, nhead, dk, dv, dim_ffn, dropout) for _ in range(num_decoder_layers)
        ])


    def forward(self, decoder_input, encoder_output, masked_self_attn_mask, cross_self_attn_mask):
        # decoder_input: [batch_size, tgt_len]
        # encoder_output: [batch_size, src_len, d_model]
        # masked_self_attn_mask: [batch_size, tgt_len, tgt_len]
        # cross_self_attn_mask: [batch_size, tgt_len, src_len]

        decoder_seq = self.decoder_we(decoder_input)  # [batch_size, tgt_len, d_model]

        decoder_seq = self.decoder_pe(decoder_seq)  # [batch_size, tgt_len, d_model]

        # 经过数个DecoderLayer
        decoder_masked_self_attn_scores = []
        decoder_cross_self_attn_scores = []
        for decoder_layer in self.decoder_layers:
            # 调用DecoderLayer
            decoder_seq, masked_score, cross_score = \
                decoder_layer(decoder_seq, encoder_output, masked_self_attn_mask, cross_self_attn_mask)  # [batch_size, src_len, d_model]
            # 将每层的自注意力分数存起来
            decoder_masked_self_attn_scores.append(masked_score)
            decoder_cross_self_attn_scores.append(cross_score)

        return decoder_seq, torch.stack(decoder_masked_self_attn_scores, dim=0), \
               torch.stack(decoder_cross_self_attn_scores, dim=0)



# 2.1
class DecoderLayer(nn.Module):
    """
    my implementation of DecoderLayer
    """
    def __init__(self, d_model: int = 512, nhead: int = 8, dk: int = 64, dv: int = 64,
                 dim_ffn: int = 2048, dropout: float = 0.1):
        """
        创建解码器层
        :param d_model: 词向量的维度
        :param nhead: 多头注意力中头的数量
        :param dk: query和key的向量维度
        :param dv: value的向量维度
        :param dim_ffn: 前馈全连接网络的中间神经元个数
        :param dropout: dropout的drop probability
        """
        super(DecoderLayer, self).__init__()

        # 初始化masked自注意力模块
        self.decoder_masked_self_attn = MultiHeadSelfAttention(d_model, nhead, dk, dv, dropout)

        # 初始化cross自注意力模块
        self.decoder_cross_self_attn = MultiHeadSelfAttention(d_model, nhead, dk, dv, dropout)

        # 初始化FFN(feedforward network)模块
        self.decoder_ffn = PositionwiseFeedForwardNetwork(d_model, dim_ffn, dropout)


    def forward(self, decoder_seq, encoder_output, masked_self_attn_mask, cross_self_attn_mask):
        # decoder_seq: [batch_size, tgt_len, d_model]
        # encoder_output: [batch_size, src_len, d_model]
        # masked_self_attn_mask: [batch_size, tgt_len, tgt_len]
        # cross_self_attn_mask: [batch_size, tgt_len, src_len]

        # masked自注意力模块
        decoder_inner_seq, masked_score = \
            self.decoder_masked_self_attn(decoder_seq, decoder_seq, decoder_seq, masked_self_attn_mask)
        # decoder_inner_seq: [batch_size, tgt_len, d_model]
        # masked_score: [batch_size, nhead, tgt_len, tgt_len]

        # crosss自注意力模块
        decoder_inner_seq, cross_score = \
            self.decoder_cross_self_attn(decoder_inner_seq, encoder_output, encoder_output, cross_self_attn_mask)
        # decoder_inner_seq: [batch_size, tgt_len, d_model]
        # cross_score: [batch_size, nhead, tgt_len, src_len]

        # FFN模块
        return self.decoder_ffn(decoder_inner_seq), masked_score, cross_score










# 源语言词典
src_dictionary = {"P": 0, "我": 1, "爱": 2, "你": 3, "今": 4, "天": 5, "气": 6, "很": 7, "好": 8}
# 目标语言词典
tgt_dictionary = {"P": 0, "i": 1, "love": 2, "you": 3, "it": 4, "is": 5, "a": 6, "fine": 7, "day": 8, "today": 9, "S": 10, "E": 11}



def make_batch(src_sentences, tgt_sentences, label_sentences):

    src_char_seq = [list(sentence) for sentence in src_sentences]
    max_src_len = len(max(src_char_seq, key=len))
    print(max_src_len)

    src_index_seq = []
    src_mask_seq = []
    for sentence_char_list in src_char_seq:
        sentence_index_list = []
        src_mask_list = []
        for i in range(max_src_len):
            if i >= len(sentence_char_list):
                sentence_index_list.append(0)
                src_mask_list.append(0)
            else:
                sentence_index_list.append(src_dictionary[sentence_char_list[i]])
                src_mask_list.append(1)
        src_index_seq.append(sentence_index_list)
        src_mask_seq.append(src_mask_list)

    src_index_seq = torch.tensor(src_index_seq)
    src_mask_seq_tensor = torch.tensor(src_mask_seq).unsqueeze(2)
    self_attn_mask = torch.bmm(src_mask_seq_tensor, src_mask_seq_tensor.transpose(-1, -2))
    self_attn_mask = (1 - self_attn_mask).to(torch.bool)

    print(src_index_seq)
    print(self_attn_mask)

    tgt_word_seq = [list(sentence.split(" ")) for sentence in tgt_sentences]
    max_tgt_len = len(max(tgt_word_seq, key=len))
    print(max_tgt_len)

    tgt_index_seq = []
    tgt_mask_seq = []
    for sentence_word_list in tgt_word_seq:
        sentence_index_list = []
        tgt_mask_list = []
        for i in range(max_tgt_len):
            if i >= len(sentence_word_list):
                sentence_index_list.append(0)
                tgt_mask_list.append(0)
            else:
                sentence_index_list.append(tgt_dictionary[sentence_word_list[i]])
                tgt_mask_list.append(1)
        tgt_index_seq.append(sentence_index_list)
        tgt_mask_seq.append(tgt_mask_list)

    tgt_index_seq = torch.tensor(tgt_index_seq)
    tgt_mask_seq_tensor = torch.tensor(tgt_mask_seq).unsqueeze(2)
    masked_self_attn_mask = torch.tril(torch.bmm(tgt_mask_seq_tensor, tgt_mask_seq_tensor.transpose(-1, -2)), diagonal=0)
    masked_self_attn_mask = (1 - masked_self_attn_mask).to(torch.bool)

    print(tgt_index_seq)
    print(masked_self_attn_mask)

    cross_self_attn_mask = torch.bmm(tgt_mask_seq_tensor, src_mask_seq_tensor.transpose(-1, -2))
    cross_self_attn_mask = (1 - cross_self_attn_mask).to(torch.bool)
    print(cross_self_attn_mask)

    return src_index_seq, tgt_index_seq, self_attn_mask, masked_self_attn_mask, cross_self_attn_mask




if __name__ == '__main__':
    # 定义数据集，batch_size = 2
    src_sentences = ["我爱你", "今天天气很好"]
    tgt_sentences = ["S i love you", "S it is a fine day today"]
    label_sentences = ["i love you E", "it is a fine day today E"]

    # 拼接成一个mini-batch,并且得到三个mask
    src_index_seq, tgt_index_seq, self_attn_mask, masked_self_attn_mask, cross_self_attn_mask = \
        make_batch(src_sentences, tgt_sentences, label_sentences)

    # 初始化模型结构
    model = Transformer(src_vocabulary_size=len(src_dictionary), tgt_vocabulary_size=len(tgt_dictionary))

    output_logits, encoder_self_attn_scores, decoder_masked_self_attn_scores, decoder_cross_self_attn_scores = \
        model(src_index_seq, tgt_index_seq, self_attn_mask, masked_self_attn_mask, cross_self_attn_mask)

    print(output_logits.shape)
    print(encoder_self_attn_scores.shape)
    print(decoder_masked_self_attn_scores.shape)
    print(decoder_cross_self_attn_scores.shape)


    # # 初始化损失函数
    # criterion = nn.CrossEntropyLoss()
    #
    # # 初始化优化器
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    # for epoch in range(20):
    #     optimizer.zero_grad()
    #     output_logits, encoder_self_attn_scores, decoder_masked_self_attn_scores, decoder_cross_self_attn_scores = \
    #         model(src_index_seq, tgt_index_seq, self_attn_mask, masked_self_attn_mask, cross_self_attn_mask)
    #     loss = criterion(outputs, target_batch.contiguous().view(-1))
    #     print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
    #     loss.backward()
    #     optimizer.step()












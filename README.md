# zang_chat
## it is just a nanochat/GPT only to please myself,several skills learned from nanochat by Carpathy


# config:12.2:
## RUSTBPE类的定义:
### 函数 def train_from-iterator:我们先继承一个类，然后在我们有的vocab_size 中去除掉特殊字符的长度，然后把我们的txt，普通字符长度，pattern放进去训练（pretrain）,得到训练过的tokenizer,然后我们找到其中的mergeable_ranks_list，在用byte()转化为字符，存到dic中，再把specail_token的dic按照顺序加在ranks的后面，再放到tiktoken的encoding中编写一下

###  函数 def encode(self,text,prepend,append,num_threads)：	•	负责把 text 编码成 token id 列表 •	text 可以是	一个字符串："hello world" 或一个字符串列表：["hello", "world"]   • prepend / append：在前/后面加一个 特殊 token（或者一个已经是 int 的 token id）  •	num_threads：批量编码时，用多少线程（tiktoken 内部会并行加速）


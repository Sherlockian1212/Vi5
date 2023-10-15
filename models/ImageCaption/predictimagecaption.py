import pandas as pd
from collections import Counter
from PIL import Image
import torch
import matplotlib.pyplot as plt
import random
import os

new_default_directory = os.getcwd()
os.chdir(new_default_directory)

def remove_single_char_word(word_list):
    lst = []
    for word in word_list:
        if len(word)>1:
            lst.append(word)

    return lst

df = pd.read_csv("captions.txt", sep=',')

df['cleaned_caption'] = df['caption'].apply(lambda caption : ['<start>'] + [word.lower() if word.isalpha() else '' for word in caption.split(" ")] + ['<end>'])
df['cleaned_caption']  = df['cleaned_caption'].apply(lambda x : remove_single_char_word(x))

df['seq_len'] = df['cleaned_caption'].apply(lambda x : len(x))
max_seq_len = df['seq_len'].max()
print(max_seq_len)

df.drop(['seq_len'], axis = 1, inplace = True)
df['cleaned_caption'] = df['cleaned_caption'].apply(lambda caption : caption + ['<pad>']*(max_seq_len-len(caption)) )

word_list = df['cleaned_caption'].apply(lambda x : " ".join(x)).str.cat(sep = ' ').split(' ')
word_dict = Counter(word_list)
word_dict =  sorted(word_dict, key=word_dict.get, reverse=True)

index_to_word = {index: word for index, word in enumerate(word_dict)}
word_to_index = {word: index for index, word in enumerate(word_dict)}

df = df.sort_values(by = 'image')
train = df.iloc[:int(0.9*len(df))]
valid = df.iloc[int(0.9*len(df)):]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = torch.load('ImageCaptioning')
start_token = word_to_index['<start>']
end_token = word_to_index['<end>']
pad_token = word_to_index['<pad>']
max_seq_len = 33
print(start_token, end_token, pad_token)

valid_img_embed = pd.read_pickle('EncodedImageValidResNet.pkl')

def generate_caption(K, img_nm):
    img_loc = 'car.jpg'
    image = Image.open(img_loc).convert("RGB")
    plt.imshow(image)

    model.eval()
    # valid_img_df = valid[valid['image']==img_nm]
    # print("Actual Caption : ")
    # print(valid_img_df['caption'].tolist())
    img_embed = valid_img_embed[img_nm].to(device)


    img_embed = img_embed.permute(0,2,3,1)
    img_embed = img_embed.view(img_embed.size(0), -1, img_embed.size(3))

    input_seq = [pad_token] * max_seq_len
    input_seq[0] = start_token

    input_seq = torch.tensor(input_seq).unsqueeze(0).to(device)
    predicted_sentence = []
    with torch.no_grad():
        for eval_iter in range(0, max_seq_len):
            output, padding_mask = model.forward(img_embed, input_seq)

            output = output[eval_iter, 0, :]

            values = torch.topk(output, K).values.tolist()
            indices = torch.topk(output, K).indices.tolist()

            next_word_index = random.choices(indices, values, k=1)[0]
            next_word = index_to_word[next_word_index]

            input_seq[:, eval_iter + 1] = next_word_index

            if next_word == '<end>':
                break

            predicted_sentence.append(next_word)
        print("\n")
        print("Predicted caption : ")
        print(" ".join(predicted_sentence + ['.']))

generate_caption(1, 'car.jpg')
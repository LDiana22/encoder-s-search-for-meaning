import matplotlib.pyplot as plt
import numpy as np
import sys
import tqdm

def _we_matrix(we_path):
  with open(we_path, encoding="utf8") as file:
    lines = file.readlines()
    file.close()
  mat = []
  word_to_idx = {}
  for _ in tqdm.tqdm(range(len(lines))):
    l = lines[_.numerator]
    word, emb = l.split()[0], l.split()[1:]
    emb = [float(x) for x in emb ]
    mat.append(emb)
    word_to_idx[word] = _.numerator
  print("Loaded we")
  return np.matrix(mat, dtype=np.float32), word_to_idx


def word_embeddings_range(we_path):
  we,_ = _we_matrix(we_path)
  return we.min(), we.max()

def plot_distributions_of_we(we_path):
  wes,_ = _we_matrix(we_path)
  colors = ['b','g','r','c','m','y','k']
  we_idx = [1,100,3254,67823,345872, 87356]
  for we,color in zip(we_idx,colors):
    w = np.array(wes[we]).reshape((1,300))[0]
    plt.plot(w, color)


  # for idx in tqdm.tqdm(range(1)):
  #   print(wes[idx.numerator])
  #   plt.plot(wes[idx.numerator])
  plt.show()


# (-3.0639, 3.2582)
# print("Glove we range:", word_embeddings_range('data/embeddings/glove.6B/glove.6B.300d.txt'))

# plot_distributions_of_we('data/embeddings/glove.6B/glove.6B.300d.txt')

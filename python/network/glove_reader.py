import os
import pickle


class GLoVe:
    def __init__(self, glove_path, words):
        self.glove_path = glove_path
        self.words = words

    def load(self, cache_name):
        print("Checking for cache")
        cache_name = '__%s__.cache' % cache_name
        if os.path.exists(cache_name):
            print("Cache found - loading from cache now!")
            with open(cache_name, 'rb') as f:
                glove_model = pickle.load(f)
        else:
            print("Cache not found")
            print("Loading GLoVe for training data")
            glove_model = dict()
            done = 0
            with open(self.glove_path) as f:
                for text in f:
                    done += 1
                    tokens = text.split()
                    if len(tokens) != 301:
                        continue

                    word = tokens[0]
                    vector = [float(tokens[x]) for x in range(1, 301)]

                    if word in self.words:
                        glove_model[word] = vector

                    if done % 1000 == 0:
                        print("%f%%" % (done / 22000))

            print("100% GLoVe data loaded")
            with open(cache_name, 'wb') as f:
                pickle.dump(glove_model, f, pickle.HIGHEST_PROTOCOL)
                print("Saved to cache")
        self.glove_model = glove_model

    def get_vector(self, word):
        if word in self.glove_model:
            return self.glove_model[word]
        else:
            return None
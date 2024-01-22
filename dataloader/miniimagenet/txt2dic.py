import pickle

if __name__ == '__main__':
    f = open("map_clsloc.txt", 'r')
    dict = {}
    while True:
        line = f.readline()
        if not line:
            break
        #print(line)
        strs = line.split(' ')
        dict[strs[0]] = strs[2][:-1]
    print(len(dict.keys()))

    fn = 'imagenet_label_textdic'
    with open(fn, 'wb') as f:
        pickle.dump(dict, f, pickle.HIGHEST_PROTOCOL)
    print('save object saved')
    f.close()



import json
from collections import defaultdict
train = defaultdict(list) 
val = defaultdict(list) 
test = defaultdict(list) 
train_classes = [0, 1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16, 19, 21,
                 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38,
                 39, 40, 41, 43, 44, 45, 46, 48, 49, 50, 52, 53, 56, 57, 58,
                 59, 61, 62, 63, 64, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
                 76, 77, 78]
val_classes = [7, 9, 17, 18, 20]
test_classes = [23, 29, 42, 47, 51, 54, 55, 60, 65, 79]

key_name_dict = {0: 'P931', 1: 'P4552', 2: 'P140', 3: 'P1923', 4: 'P150', 5: 'P6', 6: 'P27', 7: 'P449', 8: 'P1435', 9: 'P175', 10: 'P1344', 11: 'P39', 12: 'P527', 13: 'P740', 14: 'P706', 15: 'P84', 16: 'P495', 17: 'P123', 18: 'P57', 19: 'P22', 20: 'P178', 21: 'P241', 22: 'P403', 23: 'P1411', 24: 'P135', 25: 'P991', 26: 'P156', 27: 'P176', 28: 'P31', 29: 'P1877', 30: 'P102', 31: 'P1408', 32: 'P159', 33: 'P3373', 34: 'P1303', 35: 'P17', 36: 'P106', 37: 'P551', 38: 'P937', 39: 'P355', 40: 'P710', 41: 'P137', 42: 'P674', 43: 'P466', 44: 'P136', 45: 'P306', 46: 'P127', 47: 'P400', 48: 'P974', 49: 'P1346', 50: 'P460', 51: 'P86', 52: 'P118', 53: 'P264', 54: 'P750', 55: 'P58', 56: 'P3450', 57: 'P105', 58: 'P276', 59: 'P101', 60: 'P407', 61: 'P1001', 62: 'P800', 63: 'P131', 64: 'P177', 65: 'P364', 66: 'P2094', 67: 'P361', 68: 'P641', 69: 'P59', 70: 'P413', 71: 'P206', 72: 'P412', 73: 'P155', 74: 'P26', 75: 'P410', 76: 'P25', 77: 'P463', 78: 'P40', 79: 'P921'}

benchmark = "FewRel"
with open(f'{benchmark}/fewrel.json', 'r') as f:
    for line in f:
        row = json.loads(line)
        item = {
            'tokens': row['text'], 
            'h': ['', '', [[row['head'][0], row['head'][1]+1]]],
            't': ['', '', [[row['tail'][0], row['tail'][1]+1]]]
        }
        key = key_name_dict[row['label']]
        if row['label'] in train_classes:
            train[key].append(item)
        elif row['label'] in val_classes:
            val[key].append(item)
        elif row['label'] in test_classes:
            test[key].append(item)
            
with open(f'{benchmark}/train.json','w') as fout:
    fout.write(json.dumps(train))
    fout.write('\n')
with open(f'{benchmark}/val.json','w') as fout:
    fout.write(json.dumps(val))
    fout.write('\n')
with open(f'{benchmark}/test.json','w') as fout:
    fout.write(json.dumps(test))
    fout.write('\n')


import pandas as pd
import numpy as np
import random

# def text_to_csv(input_file, output_file):
#     with open(input_file, 'r') as f:
#         lines = f.readlines()[2:]
#     with open(output_file, 'w',newline='') as csv_f:
#         csv_writer = csv.writer(csv_f)
#         for i in range(0,len(lines),2):
#             if i <= 2*551:
#                 row = [lines[i].strip(),lines[i+1].strip(),1]
#             else:
#                 row = [lines[i].strip(), lines[i+1].strip(),0]
#             csv_writer.writerow(row)


# input_file = 'datasets/trainset.txt'
# output_file = 'datasets/trainset.csv'
# text_to_csv(input_file, output_file)

# input_file = 'datasets/testset.txt'
# output_file = 'datasets/testset.csv'
# text_to_csv(input_file, output_file)
# #read chemical and physical properties
#
# df = pd.read_csv('6_standardRNA.txt',sep='\t',header=0)
# df.to_csv('6_standardRNA.csv',index=False)
# table_of_chem_phy = pd.read_csv('6_standardRNA.csv')
#
# # PseDNC(physical-chemical sequence features) calculating,result for list
# class PseDNC():
#     def __init__(self,sequence):
#         self.sequence = sequence
#     #f1-f16
#     def dinucleotide_norm_frequency(self):
#         dict_key = {'GG':0,'GA':0,'GC':0,'GU':0,
#                     'AG':0,'AA':0,'AC':0,'AU':0,
#                     'CG':0,'CA':0,'CC':0,'CU':0,
#                     'UG':0,'UA':0,'UC':0,'UU':0}
#         dinucleotide_count = []
#         spilt_seq = []
#         for i in range(len(self.sequence)-2+1):
#             spilt_seq.append(self.sequence[i:i+2])
#         for key in dict_key:
#             dinucleotide_count.append(spilt_seq.count(key))
#         res_array = np.array(dinucleotide_count).reshape(-1,1)
#         norm_freq = res_array / (len(self.sequence)-1)
#         scaled_freq = norm_freq.flatten().tolist()
#
#
#         return scaled_freq
#     #the calculation of THETA
#     def THETA_calculation(self,i,j):
#         keys = ['GG', 'GA', 'GC', 'GU',
#                 'AG', 'AA', 'AC', 'AU',
#                 'CG', 'CA', 'CC', 'CU',
#                 'UG', 'UA', 'UC', 'UU']
#         THETA = 0.0
#         for key1 in keys:
#             for key2 in keys:
#                 if self.sequence[i:i+2] == key1 and self.sequence[j:j+2] == key2:
#                     THETA = (table_of_chem_phy[key1].dot(table_of_chem_phy[key1])+table_of_chem_phy[key2].dot(table_of_chem_phy[key2])-2*table_of_chem_phy[key2].dot(table_of_chem_phy[key1])) / 6.0
#
#         return THETA
#
#      #the calculation of theta
#     def theta_calculation(self,lamda):
#         L = len(self.sequence)
#         theta = 0.0
#         res = 0.0
#         for i in range(1,L-lamda):
#             res += self.THETA_calculation(i,i+lamda)
#         theta = res / float(L-1-lamda)
#         return theta
#
#     #calculation final result D=[]
#     def final_result(self,lamda,w=0.9):
#         D = [0.0]*(16+lamda)
#         sum_of_theta = 0.0
#         for i in range(1,lamda+1):
#             sum_of_theta += self.theta_calculation(i)
#         for i in range(1,17):
#             D[i-1] = self.dinucleotide_norm_frequency()[i-1] / (sum(self.dinucleotide_norm_frequency())+w*sum_of_theta)
#             #D[i-1] = float(D[i-1])
#         for i in range(17,17+lamda):
#             D[i-1] = (w*self.theta_calculation(i-16)) / (sum(self.dinucleotide_norm_frequency())+w*sum_of_theta)
#             #D[i-1] = float(D[i-1])
#
#         return D
#
# #chemical properties feature(NCP),result for array
# #encodeing A[1,1,1] G[1,0,0] C[0,1,0] U[0,0,1]
def NCP(sequence):
    L = len(sequence)
    A,G,C,U = np.array([1,1,1]),np.array([1,0,0]),np.array([0,1,0]),np.array([0,0,1])
    NCP_features = np.empty((0, 3))
    for s in sequence:
        if s == 'G':
            NCP_features = np.vstack((NCP_features,G))
        elif s == 'A':
            NCP_features = np.vstack((NCP_features, A))
        elif s == 'C':
            NCP_features = np.vstack((NCP_features, C))
        else:
            NCP_features = np.vstack((NCP_features, U))

    return NCP_features
#
# #Enhanced nucleic acid composition(ENAC),result for list
# class ENAC():
#     def __init__(self,sequence):
#         self.sequence = sequence
#     def num_of_nucleotides(self,sub_sequence):
#         L = len(sub_sequence)
#         counts = []
#         keys = {'G':0,'A':0,'C':0,'U':0}
#         for key in keys:
#             counts.append(sub_sequence.count(key))
#         return counts
#
#     def enac(self,window_size, stride):
#         L = len(self.sequence)
#         res = []
#         temp = []
#         for i in range(1, L - window_size + 1, stride):
#             if i + window_size >= L:
#                 temp = np.array(self.num_of_nucleotides(self.sequence[L-window_size+1:L+1])) / float(window_size)
#                 temp = temp.tolist()
#                 res.append(temp)
#             else:
#                 temp = np.array(self.num_of_nucleotides(self.sequence[i-1:i+window_size-1])) / float(window_size)
#                 temp = temp.tolist()
#                 res.append(temp)
#
#         res = [item for sublist in res for item in sublist]
#
#         return res
#
# def feature_fusion(sequence):
#     array1 = NCP(sequence).ravel()
#     array1 = array1.tolist()#array(201,3)
#     list1 = PseDNC(sequence)
#     list1 = list1.final_result(16) #list
#     list2 = ENAC(sequence)
#     list2 = list2.enac(5,2) #list
#     res = array1 + list1 +list2
#     return res #list


def one_hot(sequence):
    L = len(sequence)
    A, G, C, U = np.array([1, 0,0,0]), np.array([0,1, 0, 0]), np.array([0,0, 1, 0]), np.array([0,0, 0, 1])
    onehot_features = np.empty((0, 4))
    for s in sequence:
        if s == 'G':
            onehot_features = np.vstack((onehot_features, G))
        elif s == 'A':
            onehot_features = np.vstack((onehot_features, A))
        elif s == 'C':
            onehot_features = np.vstack((onehot_features, C))
        else:
            onehot_features = np.vstack((onehot_features, U))

    return onehot_features
def DPCP(sequence):
    table_of_poly = pd.read_csv('6_standardRNA.csv', header=0)
    keys = ['GG', 'GA', 'GC', 'GU',
                     'AG', 'AA', 'AC', 'AU',
                     'CG', 'CA', 'CC', 'CU',
                    'UG', 'UA', 'UC', 'UU']
    res = np.empty([6,len(sequence)])
    for i in range(len(sequence)-1):
        for key in keys:
            if sequence[i:i+2] == key:
                res[:,i] = table_of_poly[key]
    random.seed(42)
    col = random.choice(table_of_poly.columns)
    res[:,-1] = table_of_poly[col]
    return res

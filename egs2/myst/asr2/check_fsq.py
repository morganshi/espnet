import numpy as np
if __name__=="__main__":
    L=np.array([5,5,5,4,4])
    for m in range(4):
        for l in range(4):
            for k in range(5):
                for j in range(5):
                    for i in range(5):
                        h_code = np.array([i,j,k,l,m])
                        prefix_prod = np.cumprod(np.concatenate(([1], L[:-1])))
                        z_fsq = h_code[0] + np.sum(h_code[1:] * prefix_prod[1:])
                        print(z_fsq)
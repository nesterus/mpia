def reverse_dict(d):
    new_d = {}
    
    for k in d:
        new_d[d[k]] = k
        
    return new_d

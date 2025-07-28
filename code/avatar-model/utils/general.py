import os

def mkdir_ifnotexists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m

# def get_split_name(splits):
#     # example splits: [MVI_1812, MVI_1813]
#     # example output: MVI_1812+MVI_1813
#     name = ''
#     for s in splits:
#         name += str(s)
#         name += '+'
#     assert len(name) > 1
#     return name[:-1]


def get_split_name(splits):
    # example splits: [MVI_1812, MVI_1813]
    # example output: MVI_1812+MVI_1813
    '''
    원래는 ~+~+ 이런 식으로 dataset directory를 만들었으나 너무 길어지는 경우가 많아서 default로 ~to~로 변경한다.
    '''
    name = ''

    if False:
        for s in splits:
            name += str(s)
            name += '+'
        assert len(name) > 1
    else:
        # if len(name) > 255:
        if len(splits) > 1:
            name = '{}_to_{}_num_{}'.format(splits[0], splits[-1], len(splits))
            return name
        else:
            return splits[0]
    return name[:-1] # to remove the last '+'
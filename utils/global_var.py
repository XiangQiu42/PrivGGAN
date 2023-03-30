# Initialization
def _init():
    global _global_dict
    _global_dict = {}


# define a global variable
def set_value(key, value):
    _global_dict[key] = value


# get the value of a global variable
def get_value(key):
    try:
        return _global_dict[key]
    except KeyError:
        print('Filed to find' + key + '\r\n')

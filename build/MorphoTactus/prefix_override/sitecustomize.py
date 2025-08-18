import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/dev/Golain_ws/src/MorphoTactus/install/MorphoTactus'

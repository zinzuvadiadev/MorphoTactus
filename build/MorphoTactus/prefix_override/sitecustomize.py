import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/quoppo/Oak-D-Lite/MorphoTactus/install/MorphoTactus'
